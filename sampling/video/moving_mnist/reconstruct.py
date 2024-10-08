"""Tests a models ability to reconstruct the original sample."""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.transforms import v2
from typing import List

from xdiffusion.datasets.moving_mnist import MovingMNIST
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.samplers import ddim, ancestral, base
from xdiffusion.utils import (
    load_yaml,
    DotConfig,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    video_tensor_to_gif,
)

OUTPUT_NAME = "output/video/moving_mnist/reconstruct"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_model(
    config_path: str,
    num_samples: int,
    guidance: float,
    checkpoint_path: str,
    sampler: str,
    num_sampling_steps: int,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the Moving MNIST dataset.
    diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if checkpoint_path:
        diffusion_model.load_checkpoint(checkpoint_path)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    if sampler == "ddim":
        sampler = ddim.DDIMSampler()
    elif sampler == "ancestral":
        sampler = ancestral.AncestralSampler()
    else:
        raise NotImplemented(f"Sampler {sampler} not implemented.")

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        config=config,
        num_samples=num_samples,
        sampler=sampler,
        accelerator=accelerator,
        num_sampling_steps=num_sampling_steps,
    )


def sample(
    diffusion_model: DiffusionModel,
    config: DotConfig,
    sampler: base.ReverseProcessSampler,
    accelerator: Accelerator,
    num_samples: int = 64,
    num_sampling_steps: int = -1,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        assert False, "Not supported yet."

    dataset = MovingMNIST(
        ".",
        transform=v2.Compose(
            [
                # To the memory requirements, resize the MNIST
                # images from (64,64) to (32, 32).
                v2.Resize(
                    size=(config.data.image_size, config.data.image_size),
                    antialias=True,
                ),
                # Convert the motion images to (0,1) float range
                v2.ToDtype(torch.float32, scale=True),
            ]
        ),
    )
    dataloader = DataLoader(
        dataset, batch_size=num_samples, shuffle=True, num_workers=4
    )
    dataloader = accelerator.prepare(dataloader)

    # Sample a batch of videos from the dataset
    videos, classes = next(iter(dataloader))

    # Trim the number of frames to the model input
    videos = videos[:, :, : config.data.input_number_of_frames, :, :]
    prompts = convert_labels_to_prompts(classes)
    context["text_prompts"] = prompts
    context["classes"] = classes

    # Save the source video
    video_tensor_to_gif(
        videos,
        str(f"{OUTPUT_NAME}/source_sample.gif"),
    )

    # Save the first frame
    utils.save_image(
        videos[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/source_sample_first_frame.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Noise the source video to generate the initial latents
    # The images are normalized into the range (-1, 1),
    # from Section 3.3:
    # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
    x_0 = normalize_to_neg_one_to_one(videos)

    # Line 3, calculate the random timesteps for the training batch.
    # Use importance sampling here if desired.
    if diffusion_model.noise_scheduler().continuous():
        fraction = (
            num_sampling_steps / diffusion_model.noise_scheduler().steps()
            if num_sampling_steps > 0
            else 1.0
        )
        t = torch.ones((num_samples,), dtype=x_0.dtype, device=x_0.device) * fraction
    else:

        t = torch.ones((num_samples,), dtype=torch.long, device=x_0.device) * (
            num_sampling_steps
            if num_sampling_steps > 0
            else diffusion_model.noise_scheduler().steps()
        )

    # Line 4, sample from a Gaussian with mean 0 and unit variance.
    # This is the epsilon prediction target.
    epsilon = torch.randn_like(x_0)

    # Calculate forward process q_t
    z_t = diffusion_model.noise_scheduler().q_sample(x_start=x_0, t=t, noise=epsilon)

    # Sample from the model to check the quality.
    samples, intermediate_outputs = diffusion_model.sample(
        num_samples=num_samples,
        context=context,
        sampler=sampler,
        initial_noise=z_t,
        num_sampling_steps=None,  # num_sampling_steps if num_sampling_steps > 0 else None,
    )

    # Save the first frame
    utils.save_image(
        samples[:, :, 0, :, :],
        str(f"{OUTPUT_NAME}/first_frame.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the samples into an image grid
    video_tensor_to_gif(
        samples,
        str(f"{OUTPUT_NAME}/sample.gif"),
    )

    # Save the noised video
    video_tensor_to_gif(
        unnormalize_to_zero_to_one(z_t),
        str(f"{OUTPUT_NAME}/initial_noise.gif"),
    )

    # Save all of the intermediate outputs
    for i, v in enumerate(intermediate_outputs):
        video_tensor_to_gif(
            v,
            str(f"{OUTPUT_NAME}/intermediate_{i}.gif"),
        )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")


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
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_sampling_steps", type=int, default=-1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sampler", type=str, default="ancestral")
    args = parser.parse_args()

    sample_model(
        config_path=args.config_path,
        num_samples=args.num_samples,
        guidance=args.guidance,
        checkpoint_path=args.checkpoint,
        sampler=args.sampler,
        num_sampling_steps=args.num_sampling_steps,
    )


if __name__ == "__main__":
    main()
