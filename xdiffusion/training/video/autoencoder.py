from accelerate import cpu_offload, Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin
from accelerate import DistributedDataParallelKwargs
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import Callable, List, Optional

from xdiffusion import masking
from xdiffusion.datasets.utils import load_dataset
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.layers.ema import create_ema_and_scales_fn
from xdiffusion.training_utils import preprocess_training_videos
from xdiffusion.utils import (
    cycle,
    freeze,
    get_obj_from_str,
    instantiate_from_config,
    load_yaml,
    DotConfig,
    video_tensor_to_gif,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LOG_TRAINING_WEIGHTS = 100


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    dataset_name: str,
    output_path: str,
    save_and_sample_every_n: int = 1000,
    load_model_weights_from_checkpoint: str = "",
    resume_from: str = "",
    mixed_precision: str = "",
    compile: Optional[bool] = None,
    force_cpu: bool = False,
):
    """Trains an video VAE diffusion model from a config file.

    Args:
        num_training_steps: The number of training steps to train for.
        batch_size: The batch size to use in training.
        config_path: Full path to the configuration file for training.
        dataset_name: The name of the dataset to use in training.
        output_path: Full path to store output artifacts.
    """
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

    torch.autograd.set_detect_anomaly(True)
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

    # Instantiate the VAE we are training
    vae = instantiate_from_config(config.vae_config, use_config_struct=True)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        checkpoint = torch.load(load_model_weights_from_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])

    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"])

    # Show the model summary
    summary(
        vae,
        [
            (
                batch_size,
                config.data.num_channels,
                config.data.input_number_of_frames,
                config.data.image_size,
                config.data.image_size,
            )
        ],
    )

    # Now load the dataset. Do it on the main process first in case we have to download
    # it.
    with accelerator.main_process_first():
        if "training" in config and "dataset" in config.training:
            dataset, convert_labels_to_prompts = load_dataset(
                dataset_name=config.training.dataset, config=config.data, split="train"
            )

        else:
            dataset, convert_labels_to_prompts = load_dataset(
                dataset_name=dataset_name, config=config.data, split="train"
            )

    # Create the dataloader for the given dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = vae.configure_optimizers(learning_rate=4.5e-6)

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

    # Move everything to the accelerator together
    all_device_objects = accelerator.prepare(
        vae,
        dataloader,
        *optimizers,
    )
    vae = all_device_objects[0]
    dataloader = all_device_objects[1]
    optimizers = all_device_objects[2 : 2 + len(optimizers)]

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    num_samples = 64

    average_losses = [0.0 for _ in optimizers]
    average_losses_cumulative = [0.0 for _ in optimizers]
    average_posterior_mean = 0.0
    average_posterior_mean_cumulative = 0.0
    average_posterior_std = 0.0
    average_posterior_std_cumulative = 0.0

    do_compile = False
    if compile is not None:
        do_compile = compile
    else:
        if "training" in config and "compile" in config.training:
            do_compile = config.training.compile
    accelerator.print(f"Model compilation setting: {do_compile}")
    if do_compile:
        vae = torch.compile(vae)

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # All of the gradient accumulation steps count as one training step.
            for _ in range(gradient_accumulation_steps):
                # The dataset has images and classes. Let's use the classes,
                # and convert them into a fixed embedding space.
                example_data = next(dataloader)

                if len(example_data) == 2:
                    videos, classes = example_data
                    context = {"classes": classes}

                    # Convert the labels to prompts
                    prompts = convert_labels_to_prompts(classes)
                    context["text_prompts"] = prompts

                else:
                    videos, classes, context_data = example_data
                    context = {"classes": classes}
                    context.update(context_data)

                videos, masks, context = preprocess_training_videos(
                    source_videos=videos,
                    config=config,
                    context=context,
                    mask_generator=masking.IdentityMaskGenerator(),
                    batch_size=batch_size,
                    is_image_batch=False,
                )

                context["step"] = step
                context["total_steps"] = num_training_steps

                # Calculate the loss for each layer
                current_loss = []
                log_dict = {}
                with accelerator.accumulate(vae):
                    for optimizer_idx, optimizer in enumerate(optimizers):
                        with accelerator.autocast():
                            loss, reconstructions, posterior, log_dict_idx = vae(
                                batch=videos,
                                batch_idx=-1,
                                optimizer_idx=optimizer_idx,
                                global_step=step,
                            )
                            if optimizer_idx == 0:
                                optimizer_losses = loss
                            else:
                                optimizer_losses += loss
                            average_losses_cumulative[
                                optimizer_idx
                            ] += loss.detach().item()
                            current_loss.append(loss.detach().item())
                            log_dict.update(log_dict_idx)

                    # Calculate the gradients at each step in the network.
                    accelerator.backward(optimizer_losses)

                    # Clip the gradients.
                    accelerator.clip_grad_norm_(
                        vae.parameters(),
                        max_grad_norm,
                    )

                    # Step and zero the optimizer
                    for optimizer_idx, optimizer in enumerate(optimizers):
                        optimizer.step()
                        optimizer.zero_grad()

                average_posterior_mean_cumulative += posterior.mean.detach().mean()
                average_posterior_std_cumulative += posterior.std.detach().mean()

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish after all of the gradient accumulation steps.
            accelerator.wait_for_everyone()

            # Show the current loss in the progress bar.
            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {[f'{val:.4f}' for idx, val in enumerate(current_loss)]} avg_loss: {[f'{val:.4f}' for idx, val in enumerate(average_losses)]} KL: {posterior.kl().detach().mean():.4f} posterior_mean: {average_posterior_mean:.4f} posterior_std: {average_posterior_std:.4f}"
            )

            # Log all of the metrics to tensorboard
            assert len(current_loss) == 2
            tensorboard_writer.add_scalar("g_loss", current_loss[0], step)
            tensorboard_writer.add_scalar("g_avg_loss", average_losses[0], step)
            tensorboard_writer.add_scalar("d_loss", current_loss[1], step)
            tensorboard_writer.add_scalar("d_avg_loss", average_losses[1], step)
            tensorboard_writer.add_scalar("KL", posterior.kl().detach().mean(), step)
            tensorboard_writer.add_scalar(
                "Posterior Mean", posterior.mean.detach().mean().to(torch.float32), step
            )
            tensorboard_writer.add_scalar(
                "Posterior Std", posterior.std.detach().mean().to(torch.float32), step
            )
            for k, v in log_dict.items():
                tensorboard_writer.add_scalar(k, v, step)

            if step % LOG_TRAINING_WEIGHTS == 0:
                accelerator.print(log_dict)

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                average_losses = [
                    loss / save_and_sample_every_n for loss in average_losses_cumulative
                ]
                average_losses_cumulative = [0.0 for _ in optimizers]
                average_posterior_mean = (
                    average_posterior_mean_cumulative / save_and_sample_every_n
                )
                average_posterior_std = (
                    average_posterior_std_cumulative / save_and_sample_every_n
                )
                average_posterior_mean_cumulative = 0.0
                average_posterior_std_cumulative = 0.0

                # Save the first frame
                utils.save_image(
                    videos[:, :, 0, :, :],
                    str(f"{OUTPUT_NAME}/original-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )
                utils.save_image(
                    reconstructions[:, :, 0, :, :],
                    str(f"{OUTPUT_NAME}/reconstructions-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )

                # Save the samples into an image grid
                video_tensor_to_gif(
                    videos,
                    str(f"{OUTPUT_NAME}/original-{step}.gif"),
                )
                video_tensor_to_gif(
                    reconstructions,
                    str(f"{OUTPUT_NAME}/reconstructions-{step}.gif"),
                )

                # Save the images/videos to tensorbard as well
                tensorboard_writer.add_image(
                    f"samples/original-{step}",
                    utils.make_grid(
                        videos[:, :, 0, :, :], nrow=int(math.sqrt(batch_size))
                    ),
                    step,
                )
                tensorboard_writer.add_image(
                    f"samples/reconstructions-{step}",
                    utils.make_grid(
                        reconstructions[:, :, 0, :, :], nrow=int(math.sqrt(batch_size))
                    ),
                    step,
                )
                # Save a corresponding model checkpoint.
                if accelerator.is_main_process:
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": accelerator.unwrap_model(
                                vae
                            ).state_dict(),
                            "num_optimizers": len(optimizers),
                            "optimizer_state_dicts": [
                                optimizer.state_dict() for optimizer in optimizers
                            ],
                            "loss": loss,
                        },
                        f"{OUTPUT_NAME}/vae-{step}.pt",
                    )

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Update the final loss
    average_losses = [
        loss / save_and_sample_every_n for loss in average_losses_cumulative
    ]
    average_losses_cumulative = [0.0 for _ in optimizers]
    average_posterior_mean = average_posterior_mean_cumulative / save_and_sample_every_n
    average_posterior_std = average_posterior_std_cumulative / save_and_sample_every_n
    average_posterior_mean_cumulative = 0.0
    average_posterior_std_cumulative = 0.0

    # Save the first frame
    utils.save_image(
        videos[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/original-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )
    utils.save_image(
        reconstructions[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/reconstructions-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )

    # Save the samples into an image grid
    video_tensor_to_gif(
        videos,
        str(f"{OUTPUT_NAME}/original-{step}.gif"),
    )
    video_tensor_to_gif(
        reconstructions,
        str(f"{OUTPUT_NAME}/reconstructions-{step}.gif"),
    )

    # Save a corresponding model checkpoint.
    if accelerator.is_main_process:
        torch.save(
            {
                "step": step,
                "model_state_dict": accelerator.unwrap_model(vae).state_dict(),
                "num_optimizers": len(optimizers),
                "optimizer_state_dicts": [
                    optimizer.state_dict() for optimizer in optimizers
                ],
                "loss": loss,
            },
            f"{OUTPUT_NAME}/vae-{step}.pt",
        )
    accelerator.end_training()
