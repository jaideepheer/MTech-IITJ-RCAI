# M.Tech AI Final Project

> See: [Presentation.pdf](./Presentation.pdf) and [Report.pdf](./Report.pdf) for detailed information.

This work is focused at performing Neural Architecture Search (NAS) on a UNet neural network (SR3) to optimize for minimal FLOPs and inference latency; while working in a denoising diffusion generative framework (DDPM).

The task is image super-resolution on DIV2K dataset for `64x64->128x128` patches.

Broadly, this work mainly draws inspiration from the following papers,
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636v2) (UNet base model and forward diffusion)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM framework for reverce dissusion)
- [AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks](https://arxiv.org/abs/2006.08198) (FLOPs and latency addtion to loss for performing NAS)
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

