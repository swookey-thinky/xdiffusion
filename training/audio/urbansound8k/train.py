from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from einops import rearrange
import math
import os
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from tqdm import tqdm
from typing import List

from xdiffusion.datasets.urbansound8k import UrbanSound8k
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.layers.ema import create_ema_and_scales_fn
from xdiffusion.utils import (
    cycle,
    freeze,
    get_obj_from_str,
    instantiate_from_config,
    load_yaml,
    save_mel_spectrogram_audio,
    DotConfig,
)

OUTPUT_NAME = "output/audio/urbansound8k"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    sample_with_guidance: bool,
    save_and_sample_every_n: int,
    load_model_weights_from_checkpoint: str,
    resume_from: str,
    autoencoder_checkpoint: str,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Load the audio dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    dataset = UrbanSound8k(".")

    # Create the dataloader for the audio dataset
    num_samples = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create the latent encoder if it exists
    vae = None
    if "latent_encoder" in config.diffusion:
        assert autoencoder_checkpoint

        # Create and load the VAE
        vae = instantiate_from_config(
            config.diffusion.latent_encoder, use_config_struct=True
        )
        checkpoint = torch.load(autoencoder_checkpoint, map_location="cpu")
        vae.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # Freeze the VAE so that we are not updating its weights.
        vae = freeze(vae)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        diffusion_model = GaussianDiffusionCascade(config)
    elif "target" in config:
        diffusion_model = get_obj_from_str(config["target"])(config)
    else:
        diffusion_model = GaussianDiffusion_DDPM(config=config, vae=vae)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        diffusion_model.load_checkpoint(load_model_weights_from_checkpoint)

    if resume_from:
        diffusion_model.load_checkpoint(resume_from)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

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

    # Load the optimizers if we have them from the checkpoint
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

    # If there is an EMA configuration section, create it here
    if "exponential_moving_average" in config.diffusion:
        ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=config.diffusion.exponential_moving_average.target_ema_mode,
            start_ema=config.diffusion.exponential_moving_average.start_ema,
            scale_mode=config.diffusion.exponential_moving_average.scale_mode,
            start_scales=config.diffusion.exponential_moving_average.start_scales,
            end_scales=config.diffusion.exponential_moving_average.end_scales,
            total_steps=num_training_steps,
            distill_steps_per_iter=(
                config.diffusion.exponential_moving_average.distill_steps_per_iter
                if "distill_steps_per_iter"
                in config.diffusion.exponential_moving_average
                else 0
            ),
        )
    else:
        # Default scale function returns the target ema rate
        ema_scale_fn = lambda step: 0.9999, 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            audios, classes = next(dataloader)
            context = {"classes": classes}

            # Convert the labels to prompts
            prompts = convert_labels_to_prompts(classes)
            context["text_prompts"] = prompts

            # Train each cascade in the model using the given data.
            stage_loss = 0
            for model_for_layer, optimizer_for_layer, schedule_for_layer in zip(
                diffusion_model.models(), optimizers, learning_rate_schedules
            ):
                # Is this a super resolution model? If it is, then generate
                # the low resolution imagery as conditioning.
                config_for_layer = model_for_layer.config()
                context_for_layer = context.copy()
                audios_for_layer = audios

                context_for_layer["step"] = step
                context_for_layer["total_steps"] = num_training_steps

                # Convert the [B,C,M] into [B,1, C, M]
                audios_for_layer = audios_for_layer[:, None, ...]

                # Calculate the loss on the batch of training data.
                loss_dict = model_for_layer.loss_on_batch(
                    images=audios_for_layer, context=context_for_layer
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

                # Update the ema parameters for the model if they are supported
                model_for_layer.update_ema(step, num_training_steps, ema_scale_fn)

                # Reset the gradients for the next step.
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
    )
    save(diffusion_model, step, loss, optimizers, config)


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    num_samples=64,
    sample_with_guidance: bool = False,
):
    device = next(diffusion_model.parameters()).device

    context = {}

    # Sample from the model to check the quality.
    classes = torch.randint(
        0, config.data.num_classes, size=(num_samples,), device=device
    )
    prompts = convert_labels_to_prompts(classes)
    context["text_prompts"] = prompts
    context["classes"] = classes
    assert not sample_with_guidance, "Guidance sampling not supported yet"

    samples, intermediate_stage_output = diffusion_model.sample(
        num_samples=num_samples, context=context
    )

    # Save the audio samples
    base_output_path = str(f"{OUTPUT_NAME}/sample-{step}/")
    os.makedirs(base_output_path, exist_ok=True)
    output_path_spec = base_output_path + f"sample-{step}" + "-{idx}.wav"
    save_mel_spectrogram_audio(
        samples,
        output_path_spec,
    )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample-{step}.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"'{context['text_prompts'][i]}' ")


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
        ("air conditioner", "AC"),
        ("car horn", "horn"),
        ("children playing", "children"),
        ("dog bark", "barking"),
        ("drilling", "drill"),
        ("engine idling", "car engine"),
        ("gun shot", "gun"),
        ("jackhammer", "hammering"),
        ("siren", "alarm"),
        ("street music", "music"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        text_labels[labels[i]][torch.randint(0, len(text_labels[labels[i]]), size=())]
        for i in range(labels.shape[0])
    ]
    return prompts


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--sample_with_guidance", action="store_true")
    parser.add_argument("--save_and_sample_every_n", type=int, default=1000)
    parser.add_argument("--load_model_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--autoencoder_checkpoint", type=str, default="")
    args = parser.parse_args()

    train(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        sample_with_guidance=args.sample_with_guidance,
        save_and_sample_every_n=args.save_and_sample_every_n,
        load_model_weights_from_checkpoint=args.load_model_weights_from_checkpoint,
        resume_from=args.resume_from,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
    )


if __name__ == "__main__":
    main()
