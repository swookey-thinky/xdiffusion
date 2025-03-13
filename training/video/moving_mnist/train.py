from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import utils as torchvision_utils
from torchvision.transforms import v2
from tqdm import tqdm
from typing import List

from xdiffusion.utils import (
    load_yaml,
    cycle,
    DotConfig,
    video_tensor_to_gif,
    normalize_to_neg_one_to_one,
)
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.datasets.moving_mnist import MovingMNIST
from xdiffusion.training_utils import get_training_batch, preprocess_training_videos
from xdiffusion import masking

OUTPUT_NAME = "output/video/moving_mnist"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    save_and_sample_every_n: int,
    load_model_weights_from_checkpoint: str,
    resume_from: str,
    sample_with_guidance: bool = False,
    joint_image_video_training_step: int = -1,
    force_cpu: bool = False,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Load the MNIST dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    dataset = MovingMNIST(
        ".",
        transform=v2.Compose(
            [
                # To the memory requirements, resize the MNIST
                # videos from (64,64) to (32, 32).
                v2.Resize(
                    size=(config.data.image_size, config.data.image_size),
                    antialias=True,
                ),
                # Convert the motion videos to (0,1) float range
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Technically this is not correct, as the val set is the same as the train set
    # TODO: Create a real validation set for Moving MNIST.
    num_samples = 16
    validation_dataloader = DataLoader(
        dataset, batch_size=num_samples, shuffle=True, num_workers=4
    )

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        diffusion_model = GaussianDiffusionCascade(config=config)
    else:
        diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        diffusion_model.load_checkpoint(load_model_weights_from_checkpoint)

    if resume_from:
        diffusion_model.load_checkpoint(resume_from)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator_force_cpu = True if force_cpu else None
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
        cpu=accelerator_force_cpu,
    )

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader, validation_dataloader = accelerator.prepare(
        dataloader, validation_dataloader
    )

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = diffusion_model.configure_optimizers(learning_rate=2e-4)

    # Step counter to keep track of training
    step = 0

    # Load the optimizers and step counter if we have them from the checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        num_optimizers = checkpoint["num_optimizers"]
        for i in range(num_optimizers):
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dicts"][i])
        
        if "step" in checkpoint:
            step = checkpoint["step"]            

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)
    for optimizer_idx in range(len(optimizers)):
        optimizers[optimizer_idx] = accelerator.prepare(optimizers[optimizer_idx])

    # Configure the learning rate schedule
    learning_rate_schedules = diffusion_model.configure_learning_rate_schedule(
        optimizers
    )
    for schedule_idx in range(len(learning_rate_schedules)):
        learning_rate_schedules[schedule_idx] = accelerator.prepare(
            learning_rate_schedules[schedule_idx]
        )

    # Create a mask generation strategy for each model (if a cascade)
    mask_generators = []
    for model in diffusion_model.models():
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

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
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
                    diffusion_model.models(),
                    optimizers,
                    learning_rate_schedules,
                    mask_generators,
                )
            ):
                # Is this a super resolution model? If it is, then generate
                # the low resolution imagery as conditioning.
                config_for_layer = model_for_layer.config()
                context_for_layer = context.copy()

                # Preprocess the training videos (e.g. clip or skip frames to match the setup)
                videos_for_layer, mask_for_layer, context = preprocess_training_videos(
                    source_videos=source_videos,
                    config=config_for_layer,
                    context=context_for_layer,
                    mask_generator=mask_generator,
                    batch_size=batch_size,
                    is_image_batch=is_image_batch,
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
                loss_dict = model_for_layer.loss_on_batch(
                    images=videos_for_layer,
                    context=context_for_layer,
                    stage_idx=stage_idx,
                )
                loss = loss_dict["loss"]

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

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

            # Show the current loss in the progress bar.
            stage_loss = stage_loss / len(optimizers)
            progress_bar.set_description(
                f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += stage_loss

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=diffusion_model,
                    step=step,
                    config=config,
                    num_samples=num_samples,
                    sample_with_guidance=sample_with_guidance,
                    validation_dataloader=validation_dataloader,
                )
                save(diffusion_model, step, loss, optimizers, config)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        sample_with_guidance=sample_with_guidance,
        validation_dataloader=validation_dataloader,
    )
    save(diffusion_model, step, loss, optimizers, config)


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    validation_dataloader,
    num_samples=64,
    sample_with_guidance: bool = False,
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
            str(f"{OUTPUT_NAME}/low_resolution_context-{step}.gif"),
        )
        video_tensor_to_gif(
            source_videos,
            str(f"{OUTPUT_NAME}/low_resolution_source-{step}.gif"),
        )
        video_tensor_to_gif(
            videos,
            str(f"{OUTPUT_NAME}/low_resolution_preprocessed-{step}.gif"),
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
                str(f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}.gif"),
            )

            # Save the intermedidate stages if they exist
            if intermediate_stage_output is not None:
                for layer_idx, intermediate_output in enumerate(
                    intermediate_stage_output
                ):
                    video_tensor_to_gif(
                        intermediate_output,
                        str(
                            f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}-stage-{layer_idx}.gif"
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
            str(f"{OUTPUT_NAME}/sample-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )

        # Save the samples into an image grid
        video_tensor_to_gif(
            samples,
            str(f"{OUTPUT_NAME}/sample-{step}.gif"),
        )

        # Save the intermedidate stages if they exist
        if intermediate_stage_output is not None:
            for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
                video_tensor_to_gif(
                    intermediate_output,
                    str(f"{OUTPUT_NAME}/sample-{step}-stage-{layer_idx}.gif"),
                )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample-{step}.txt", "w") as fp:
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
        f"{OUTPUT_NAME}/diffusion-{step}.pt",
    )


def convert_labels_to_prompts(labels: torch.Tensor) -> List[str]:
    """Converts MNIST class labels to text prompts.

    Supports both the strings "0" and "zero" to describe the
    class labels.
    """
    # The conditioning we pass to the model will be a vectorized-form of
    # MNIST classes. Since we have a fixed number of classes, we can create
    # a hard-coded "embedding" of the MNIST class label.
    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        f"{text_labels[labels[i][0]][torch.randint(0, len(text_labels[labels[i][0]]), size=())]} and {text_labels[labels[i][1]][torch.randint(0, len(text_labels[labels[i][1]]), size=())]}"
        for i in range(labels.shape[0])
    ]
    return prompts


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--save_and_sample_every_n", type=int, default=10000)
    parser.add_argument("--load_model_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--joint_image_video_training_step", type=int, default=-1)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()

    train(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        save_and_sample_every_n=args.save_and_sample_every_n,
        load_model_weights_from_checkpoint=args.load_model_weights_from_checkpoint,
        resume_from=args.resume_from,
        joint_image_video_training_step=args.joint_image_video_training_step,
        force_cpu=args.force_cpu,
    )


if __name__ == "__main__":
    main()
