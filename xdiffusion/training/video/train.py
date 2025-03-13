from accelerate import (
    Accelerator,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
)
from accelerate.utils import GradientAccumulationPlugin
from datetime import datetime
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as torchvision_utils
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Callable, List, Optional

from xdiffusion import masking
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.datasets.utils import load_dataset
from xdiffusion.training_utils import get_training_batch, preprocess_training_videos
from xdiffusion.utils import (
    load_yaml,
    cycle,
    DotConfig,
    video_tensor_to_gif,
    normalize_to_neg_one_to_one,
    instantiate_from_config,
    get_obj_from_str,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    output_path: str,
    dataset_name: str,
    save_and_sample_every_n: int,
    load_model_weights_from_checkpoint: str,
    load_vae_weights_from_checkpoint: str,
    resume_from: str,
    sample_with_guidance: bool = False,
    joint_image_video_training_step: int = -1,
    mixed_precision: str = "",
    force_cpu: bool = False,
    compile: Optional[bool] = None,
):
    # Open the model configuration
    config = load_yaml(config_path)

    if "training" in config and "dataset" in config.training:
        dataset_name = config.training.dataset
    else:
        if not dataset_name:
            raise ValueError(
                "--dataset_name must be passed if there is no dataset specified in the config file."
            )

        # Use the batch size from the training path, unless overridden by the command line
    if "training" in config and "batch_size" in config.training:
        if batch_size <= 0:
            # Only override if its not on the command line
            batch_size = config.training.batch_size
    else:
        # It's not in the config file, make sure its specified
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be specified in the configuration file or on the command line with --batch_size"
            )

    if "training" in config and "num_training_steps" in config.training:
        if num_training_steps <= 0:
            # Only override if this is not on the command line
            num_training_steps = config.training.num_training_steps
    else:
        # If its not in the config file, make sure it was specified
        if num_training_steps <= 0:
            raise ValueError(
                "The number of training steps (num_training_steps) must be specified on the command line or in the config file."
            )

    OUTPUT_NAME = f"{output_path}/{dataset_name}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Create the tensorboard summary to log the training progress.
    tensorboard_writer = SummaryWriter(
        os.path.join(
            os.path.join(OUTPUT_NAME, "tensorboard"),
            datetime.now().strftime("%Y%m%d%H%M%S"),
        )
    )

    # Check to see if we are using gradient accumulation
    gradient_accumulation_steps = 1
    if "training" in config and "gradient_accumulation_steps" in config.training:
        gradient_accumulation_steps = config.training.gradient_accumulation_steps

    if "training" in config and not mixed_precision:
        mixed_precision = config.training.mixed_precision
    else:
        if not mixed_precision:
            mixed_precision = "no"

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False, broadcast_buffers=False
    )

    # The accelerate library will handle of the GPU device management for us.
    # Make sure to create it early so that we can gate some data loading on it.
    accelerator_force_cpu = True if force_cpu else None
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision=mixed_precision,
        gradient_accumulation_plugin=(
            GradientAccumulationPlugin(
                num_steps=gradient_accumulation_steps,
                adjust_scheduler=True,
                sync_with_dataloader=False,
            )
            if gradient_accumulation_steps > 1
            else None
        ),
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
        cpu=accelerator_force_cpu,
    )
    accelerator.print(
        f"Training with {gradient_accumulation_steps} gradient accumulation steps."
    )
    accelerator.print(f"Training with mixed precision: {mixed_precision}.")

    # Create the VAE, if it exists
    vae = None
    if "latent_encoder" in config.diffusion:
        vae = instantiate_from_config(
            config.diffusion.latent_encoder, use_config_struct=True
        )

        if load_vae_weights_from_checkpoint:
            ckpt = torch.load(load_vae_weights_from_checkpoint, map_location="cpu")[
                "model_state_dict"
            ]

            # Remove "module." from the keys
            keys = list(ckpt.keys())
            sd = {}
            for k in keys:
                if k.startswith("module."):
                    sd[k[7:]] = ckpt[k]
                else:
                    sd[k] = ckpt[k]

            vae.load_state_dict(sd, strict=True)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        source_diffusion_model = GaussianDiffusionCascade(config=config, vae=vae)
    elif "target" in config:
        source_diffusion_model = get_obj_from_str(config["target"])(
            config=config, vae=vae
        )
    else:
        source_diffusion_model = GaussianDiffusion_DDPM(config=config, vae=vae)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        source_diffusion_model.load_checkpoint(load_model_weights_from_checkpoint)

    if resume_from:
        source_diffusion_model.load_checkpoint(resume_from)

    # Build context to display the model summary.
    source_diffusion_model.print_model_summary()

    # Now load the dataset. Do it on the main process first in case we have to download
    # it.
    with accelerator.main_process_first():
        dataset, convert_labels_to_prompts = load_dataset(
            dataset_name=dataset_name, config=config.data, split="train"
        )
        # Technically this is not correct, as the val set is the same as the train set
        # TODO: Create a real validation set for Moving MNIST.
        validation_dataset, _ = load_dataset(
            dataset_name=dataset_name, config=config.data, split="train"
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    num_samples = 16
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=num_samples, shuffle=False, num_workers=1
    )

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = source_diffusion_model.configure_optimizers(learning_rate=2e-4)

    # step counter to keep track of training
    step = 0
    
    # Load the optimizers and step counter if we have them from the checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        num_optimizers = checkpoint["num_optimizers"]
        for i in range(num_optimizers):
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dicts"][i])

        if "step" in checkpoint:
            step = checkpoint["step"]

    # Configure the learning rate schedule
    learning_rate_schedules = source_diffusion_model.configure_learning_rate_schedule(
        optimizers
    )

    # Move everything to the accelerator together. We are going to expand all of
    # the layers of the diffusion model (if we are a cascade, otherwise its singular)
    # as we send them to the accelerator.
    all_device_objects = accelerator.prepare(
        dataloader,
        validation_dataloader,
        *source_diffusion_model.models(),
        *optimizers,
        *learning_rate_schedules,
    )

    num_models = len(optimizers)
    dataloader = all_device_objects[0]
    validation_dataloader = all_device_objects[1]
    diffusion_models = all_device_objects[2 : 2 + num_models]
    optimizers = all_device_objects[2 + num_models : 2 + 2 * num_models]
    learning_rate_schedules = all_device_objects[2 + 2 * num_models :]
    assert len(optimizers) == len(
        learning_rate_schedules
    ), "Optimizers and learning rate schedules are not the same length!"

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Create a mask generation strategy for each model (if a cascade)
    mask_generators = []
    for model in source_diffusion_model.models():
        if "training" in model.config() and "mask_ratios" in model.config().training:
            mask_generators.append(
                masking.OpenSoraMaskGenerator(
                    mask_ratios=model.config().training.mask_ratios.to_dict()
                )
            )
        else:
            mask_generators.append(masking.IdentityMaskGenerator())

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    do_compile = False
    if compile is not None:
        do_compile = compile
    else:
        if "training" in config and "compile" in config.training:
            do_compile = config.training.compile
    accelerator.print(f"Model compilation setting: {do_compile}")
    if do_compile:
        compiled_models = []
        for model in diffusion_models:
            compiled_models.append(torch.compile(model))
        diffusion_models = compiled_models

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # All of the gradient accumulation steps count as one training step.
            for _ in range(gradient_accumulation_steps):
                with accelerator.accumulate(diffusion_models):
                    # The dataset has videos and classes. Let's use the classes,
                    # and convert them into a fixed embedding space.
                    is_image_batch = (
                        joint_image_video_training_step > 1
                        and (step % joint_image_video_training_step) == 0
                    ) or joint_image_video_training_step == 1

                    source_videos, labels = get_training_batch(
                        dataloader,
                        is_image_batch=is_image_batch,
                    )
                    context = {"labels": labels}
                    context["is_image_batch"] = is_image_batch

                    # Convert the labels to text prompts
                    text_prompts = convert_labels_to_prompts(labels)
                    context["text_prompts"] = text_prompts

                    # Train each cascade in the model using the given data.
                    stage_loss = 0
                    for stage_idx, (
                        model_for_layer,
                        optimizer_for_layer,
                        schedule_for_layer,
                        mask_generator,
                    ) in enumerate(
                        zip(
                            diffusion_models,
                            optimizers,
                            learning_rate_schedules,
                            mask_generators,
                        )
                    ):
                        # Is this a super resolution model? If it is, then generate
                        # the low resolution imagery as conditioning.
                        config_for_layer = accelerator.unwrap_model(
                            model_for_layer
                        ).config()
                        context_for_layer = context.copy()

                        # Preprocess the training videos (e.g. clip or skip frames to match the setup)
                        videos_for_layer, mask_for_layer, context = (
                            preprocess_training_videos(
                                source_videos=source_videos,
                                config=config_for_layer,
                                context=context_for_layer,
                                mask_generator=mask_generator,
                                batch_size=batch_size,
                                is_image_batch=is_image_batch,
                            )
                        )
                        context_for_layer["video_mask"] = mask_for_layer

                        # Make sure the text prompts are are the same length as the batch size
                        # after preprocessing
                        if len(context["text_prompts"]) > videos_for_layer.shape[0]:
                            context["text_prompts"] = context["text_prompts"][
                                : videos_for_layer.shape[0]
                            ]

                        if "super_resolution" in config_for_layer:
                            context_for_layer = _add_low_resolution_context(
                                context=context_for_layer,
                                config=config_for_layer,
                                training_videos=videos_for_layer,
                                source_videos=source_videos,
                            )

                        # Calculate the loss on the batch of training data.
                        with accelerator.autocast():
                            loss_dict = model_for_layer(
                                images=videos_for_layer,
                                context=context_for_layer,
                                stage_idx=stage_idx,
                            )
                            loss = loss_dict["loss"]

                        # Calculate the gradients at each step in the network.
                        accelerator.backward(loss)

                        # Clip the gradients.
                        accelerator.clip_grad_norm_(
                            model_for_layer.parameters(),
                            max_grad_norm,
                        )

                        # Perform the gradient descent step using the optimizer.
                        optimizer_for_layer.step()
                        schedule_for_layer.step()

                        # Resent the gradients for the next step.
                        optimizer_for_layer.zero_grad()
                        stage_loss += loss.item()

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Show the current loss in the progress bar.
            stage_loss = stage_loss / len(optimizers)
            progress_bar.set_description(
                f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += stage_loss

            tensorboard_writer.add_scalar("loss", stage_loss, step)

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=source_diffusion_model,
                    step=step,
                    config=config,
                    num_samples=num_samples,
                    sample_with_guidance=sample_with_guidance,
                    validation_dataloader=validation_dataloader,
                    output_path=OUTPUT_NAME,
                    convert_labels_to_prompts=convert_labels_to_prompts,
                    tensorboard_writer=tensorboard_writer,
                )
                if accelerator.is_main_process:
                    save(
                        source_diffusion_model,
                        step,
                        loss,
                        optimizers,
                        config,
                        output_path=OUTPUT_NAME,
                    )
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=source_diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        sample_with_guidance=sample_with_guidance,
        validation_dataloader=validation_dataloader,
        output_path=OUTPUT_NAME,
        convert_labels_to_prompts=convert_labels_to_prompts,
        tensorboard_writer=tensorboard_writer,
    )
    if accelerator.is_main_process:
        save(
            source_diffusion_model,
            step,
            loss,
            optimizers,
            config,
            output_path=OUTPUT_NAME,
        )
    accelerator.end_training()


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    validation_dataloader,
    output_path: str,
    convert_labels_to_prompts: Callable[[torch.Tensor], List[str]],
    num_samples=64,
    sample_with_guidance: bool = False,
    tensorboard_writer=None,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        source_videos, classes = next(iter(validation_dataloader))
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

        videos, context = preprocess_training_videos(
            source_videos=source_videos, config=config, context=context
        )
        context = _add_low_resolution_context(
            context=context,
            config=config,
            training_videos=videos,
            source_videos=source_videos,
        )

        # Save the low-resolution imagery if it was used.
        video_tensor_to_gif(
            context[config.super_resolution.conditioning_key],
            str(f"{output_path}/low_resolution_context-{step}.gif"),
        )
        video_tensor_to_gif(
            source_videos,
            str(f"{output_path}/low_resolution_source-{step}.gif"),
        )
        video_tensor_to_gif(
            videos,
            str(f"{output_path}/low_resolution_preprocessed-{step}.gif"),
        )
    else:
        # Sample from the model to check the quality.
        classes = torch.randint(
            0, config.data.num_classes, size=(num_samples, 2), device=device
        )
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

    if sample_with_guidance:
        for guidance in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0]:
            samples, intermediate_stage_output = diffusion_model.sample(
                num_samples=num_samples,
                context=context,
                classifier_free_guidance=guidance,
            )

            # Save the samples into an image grid
            video_tensor_to_gif(
                samples,
                str(f"{output_path}/sample-{step}-cfg-{guidance}.gif"),
            )

            # Save the intermedidate stages if they exist
            if intermediate_stage_output is not None:
                for layer_idx, intermediate_output in enumerate(
                    intermediate_stage_output
                ):
                    video_tensor_to_gif(
                        intermediate_output,
                        str(
                            f"{output_path}/sample-{step}-cfg-{guidance}-stage-{layer_idx}.gif"
                        ),
                    )

    else:
        # Add the flexible diffusion modeling context
        B, C, F, H, W = (
            num_samples,
            config.data.num_channels,
            config.data.input_number_of_frames,
            config.data.image_size,
            config.data.image_size,
        )

        context["frame_indices"] = torch.tile(
            torch.arange(end=F, device=device)[None, ...],
            (B, 1),
        )
        context["observed_mask"] = torch.zeros(
            size=(B, C, F, 1, 1), dtype=torch.float32, device=device
        )
        context["latent_mask"] = torch.ones(
            size=(B, C, F, 1, 1), dtype=torch.float32, device=device
        )

        if "latent_encoder" in config.diffusion.to_dict():
            context["x0"] = normalize_to_neg_one_to_one(
                torch.zeros(
                    size=(
                        B,
                        config.diffusion.score_network.params.input_channels,
                        config.diffusion.score_network.params.input_number_of_frames,
                        config.diffusion.score_network.params.input_spatial_size,
                        config.diffusion.score_network.params.input_spatial_size,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            )
            context["video_mask"] = torch.ones(
                size=(B, config.diffusion.score_network.params.input_number_of_frames),
                dtype=torch.bool,
                device=device,
            )
        else:
            context["x0"] = normalize_to_neg_one_to_one(
                torch.zeros(size=(B, C, F, H, W), dtype=torch.float32, device=device)
            )
            context["video_mask"] = torch.ones(
                size=(B, F), dtype=torch.bool, device=device
            )

        samples, intermediate_stage_output = diffusion_model.sample(
            num_samples=num_samples, context=context
        )

        # Save the first frame
        torchvision_utils.save_image(
            samples[:, :, 0, :, :],
            str(f"{output_path}/sample-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )

        # Save the samples into an image grid
        video_tensor_to_gif(
            samples,
            str(f"{output_path}/sample-{step}.gif"),
        )

        if tensorboard_writer is not None:
            tensorboard_writer.add_image(
                f"samples/first_frame-{step}",
                torchvision_utils.make_grid(
                    samples[:, :, 0, :, :], nrow=int(math.sqrt(num_samples))
                ),
                step,
            )

        # Save the intermedidate stages if they exist
        if intermediate_stage_output is not None:
            for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
                video_tensor_to_gif(
                    intermediate_output,
                    str(f"{output_path}/sample-{step}-stage-{layer_idx}.gif"),
                )

    # Save the prompts that were used
    with open(f"{output_path}/sample-{step}.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")


def _add_low_resolution_context(context, config, training_videos, source_videos):
    # First create the low resolution context.
    if config.super_resolution.is_spatial:
        low_resolution_spatial_size = config.super_resolution.low_resolution_size
        low_resolution_videos = v2.functional.resize(
            training_videos,
            size=(
                low_resolution_spatial_size,
                low_resolution_spatial_size,
            ),
            antialias=True,
        )
        context[config.super_resolution.conditioning_key] = low_resolution_videos
    elif config.super_resolution.is_temporal:
        # Make sure the source videos are the same spatial size as the training videos
        # and model input.
        B, C, F, H, W = source_videos.shape
        if H != config.data.image_size and W != config.data.image_size:
            source_videos = v2.functional.resize(
                source_videos,
                size=(
                    config.data.image_size,
                    config.data.image_size,
                ),
                antialias=True,
            )

        # Generate the low-resolution temporal videos, making sure to
        # match the frame interpolation here with the frame sampling from
        # the training data.
        frameskip_method = config.super_resolution.low_resolution_sampling_scheme
        assert frameskip_method.startswith("frameskip")
        frameskip = int(frameskip_method.split("_")[1])
        frame_indices = list(
            range(
                0,
                frameskip * config.super_resolution.low_resolution_size,
                frameskip,
            )
        )
        low_resolution_videos = source_videos[:, :, frame_indices, :, :]
        context[config.super_resolution.conditioning_key] = low_resolution_videos
    else:
        raise NotImplementedError("Super resolution layer is not spatial or temporal!")
    return context


def save(
    diffusion_model,
    step,
    loss,
    optimizers: List[torch.optim.Optimizer],
    config: DotConfig,
    output_path: str,
):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "num_optimizers": len(optimizers),
            "optimizer_state_dicts": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "loss": loss,
            "config": config.to_dict(),
        },
        f"{output_path}/diffusion-{step}.pt",
    )
