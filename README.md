# xdiffusion
A unified media (Image, Video, Audio, Text) diffusion repository, for education and learning.

## Requirements

This package built using PyTorch and written in Python 3. To setup an environment to run all of the lessons, we suggest using conda or venv:

```
> python3 -m venv xdiffusion_env
> source xdiffusion_env/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run from the root of the repository, and you should set your python path to include the repository root:

```
> export PYTHONPATH=$(pwd)
```

If you have issues with PyTorch and different CUDA versions on your instance, make sure to install the correct version of PyTorch for the CUDA version on your machine. For example, if you have CUDA 11.8 installed, you can install PyTorch using:

```
> pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Image Diffusion

### Training Datasets

### Image Models
The following is a list of the supported image models, their current results, and a link to their configuration files and documentation.

| Date  | Name  | Paper | Config | Results | Instructions
| :---- | :---- | ----- | ------ | ----- | -----

## Video Diffusion

### Training Datasets

### Video Diffusion Models
The following is a list of the supported image models, their current results, and a link to their configuration files and documentation.

| Date  | Name  | Paper | Config | Results | Instructions
| :---- | :---- | ----- | ------ | ----- | -----
| April 2022 | Video Diffusion Models | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/video_diffusion_models.yaml) | ![Video Diffusion Models](https://drive.google.com/uc?export=view&id=1pF6WVY8_dlGudxZIsml3VWPxbUs0ONfa) | [instructions](https://github.com/swookey-thinky/xdiffusion/blob/main/docs/video/video_diffusion_models.md)
| May 2022 | CogVideo | [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868) | | |
| May 2022 | FDM | [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495) | [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/flexible_diffusion_modeling.yaml) | ![Flexible Diffusion Models](https://drive.google.com/uc?export=view&id=1B2raR3_suRf8qAUP4jzi8YIwka-UHwrU) | [instructions](https://github.com/swookey-thinky/xdiffusion/blob/main/docs/video/flexible_diffusion_modeling.md)
| September 2022 | Make-A-Video | [Make-A-Video: Text-to-Video Generation without Text-Video Data](https://arxiv.org/abs/2209.14792) | [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/make_a_video.yaml) |  ![Make-A-Video](https://drive.google.com/uc?export=view&id=1dm4H7lsliib4KW-4T4DJeiFLRi2Ph2JD) | [instructions](https://github.com/swookey-thinky/xdiffusion/blob/main/docs/video/make_a_video.md)
| October 2022 | Imagen Video | [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303) | [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/imagen_video.yaml) |  ![Imagen Video](https://drive.google.com/uc?export=view&id=1TKwiYQYnIZZ8fM5juQMEeJkc6clPkCvV) | [instructions](https://github.com/swookey-thinky/xdiffusion/blob/main/docs/video/imagen_video.md)
| October 2022 | Phenaki | [Phenaki: Variable Length Video Generation From Open Domain Textual Description](https://arxiv.org/abs/2210.02399) | | |
| December 2022 | Tune-A-Video  | [Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565) | | |
| February 2023 | | [Structure and Content-Guided Video Synthesis with Diffusion Models](https://arxiv.org/abs/2302.03011) | | |
| March 2023 | Text2Video-Zero | [Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://arxiv.org/abs/2303.13439) | | |
| April 2023 | Video LDM | [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818) | | |
| July 2023 | AnimateDiff | [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725) | | |
| August 2023 | ModelScopeT2V | [ModelScope Text-to-Video Technical Report](https://arxiv.org/abs/2308.06571) | | |
| September 2023 | Show-1 | [Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2309.15818) | | |
| September 2023 | LaVie | [LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models](https://arxiv.org/abs/2309.15103) | | |
| October 2023 | VideoCrafter 1 | [VideoCrafter1: Open Diffusion Models for High-Quality Video Generation](https://arxiv.org/abs/2310.19512) | | |
| November 2023 | Emu Video | [Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning](https://arxiv.org/abs/2311.10709) | | |
| November 2023 | | [Decouple Content and Motion for Conditional Image-to-Video Generation](https://arxiv.org/abs/2311.14294) | | |
| November 2023 | Stable Video Diffusion | [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127) | | |
| December 2023 | VideoBooth | [VideoBooth: Diffusion-based Video Generation with Image Prompts](https://arxiv.org/abs/2312.00777) | | |
| December 2023 | LivePhoto | [LivePhoto: Real Image Animation with Text-guided Motion Control](https://arxiv.org/abs/2312.02928) | | |
| December 2023 | HiGen | [Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation](https://arxiv.org/abs/2312.04483) | | |
| December 2023 | AnimateZero | [AnimateZero: Video Diffusion Models are Zero-Shot Image Animators](https://arxiv.org/abs/2312.03793) | | |
| December 2023 | W.A.L.T | [Photorealistic Video Generation with Diffusion Models](https://arxiv.org/abs/2312.06662) | | |
| December 2023 | VideoLCM | [VideoLCM: Video Latent Consistency Model](https://arxiv.org/abs/2312.09109) | | |
| December 2023 | GenTron | [GenTron: Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2312.04557) | | |
| January 2024 | Latte | [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048) | | |
| January 2024 | VideoCrafter 2 | [VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models](https://arxiv.org/abs/2401.09047) | | |
| January 2024 | Lumiere | [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945) | | |
| February 2024 | AnimateLCM | [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning](https://arxiv.org/abs/2402.00769) | | |
| February 2024 | Video-LaVIT | [Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization](https://arxiv.org/abs/2402.03161) | | |
| February 2024 | Snap Video | [Snap Video: Scaled Spatiotemporal Transformers for Text-to-Video Synthesis](https://arxiv.org/abs/2402.14797) | | |
| February 2024 | SORA | [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) [alternative](https://arxiv.org/abs/2402.17177)| | |
| April 2024 | TI2V-Zero | [TI2V-Zero: Zero-Shot Image Conditioning for Text-to-Video Diffusion Models](https://arxiv.org/abs/2404.16306) | | |
| May 2024 | Vidu | [Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models](https://arxiv.org/abs/2405.04233) | | |
| May 2024 | FIFO-Diffusion | [FIFO-Diffusion: Generating Infinite Videos from Text without Training](https://arxiv.org/abs/2405.11473) | | |

## TODO

- [ ] Port the image diffusion repository
- [ ] Port the video diffusion respository
- [ ] Unify all of the different attention mechanisms under Transformer based (B, L, D) and pixel based (B, C, H, W).