# Denoising Diffusion Probabilistic Models (DDPMs)

Implementation, derivation, and experimental analysis of DDPMs on MNIST and CIFAR 10, built from scratch in PyTorch.

**Authors:** Hoang Dung Vu Minh and Christopher Won

## Overview

Denoising Diffusion Probabilistic Models are a class of deep generative models: neural networks trained to synthesize new data that closely resembles a training distribution. Introduced by Sohl Dickstein et al. (2015) and popularized by Ho, Jain, and Abbeel (2020), DDPMs are known for stable training and strong sample quality, often matching or surpassing GANs without any adversarial objective.

The method has two phases:

1. **Forward diffusion process** — Gaussian noise is added to an image over many timesteps until the original signal is destroyed and only noise remains.
2. **Reverse process** — a neural network (a U-Net in this project) learns to iteratively remove that noise, so that starting from pure random noise, it can generate a new, coherent sample.

This repository derives the full DDPM training objective from first principles (the evidence lower bound and its closed form simplification), then implements and trains the forward and reverse processes from scratch, without relying on any prebuilt diffusion library.

## Repository Contents

| File | Description |
|---|---|
| `DDPM-REPORT.pdf` | Full written report: theoretical derivation of the DDPM objective, methodology, results, architectural discussion, and proofs |
| `DDPM_SLIDES.pdf` | Presentation slides summarizing the project |
| `MNIST_Diffusion_model_1_reproducible.ipynb` | Baseline U-Net trained on MNIST |
| `MNIST_Diffusion_model_2_reproducible.ipynb` | Improved U-Net on MNIST, adds an extra encoder/decoder block, a self attention mechanism, and transformer style time embeddings |
| `CIFAR_10_Diffusion_model_1_reproducible.ipynb` | The MNIST model 2 architecture adapted to RGB input, applied directly to CIFAR 10 |
| `CIFAR_10_Diffusion_model_2_reproducible.ipynb` | CIFAR 10 model with Group Normalization added to each encoder/decoder block and additional self attention layers in the encoder path |
| `CIFAR_10_Diffusion_model_3_reproducible.ipynb` | The most refined CIFAR 10 model; also includes code to produce a progressive generation grid (noise to final sample), a result not shown in the written report |

Each notebook is self contained and reproducible: it includes a Colab badge and runs end to end (data loading, training, and sample generation) without depending on the other notebooks.

## Method Summary

The forward process turns an image into noise through a fixed Markov chain of Gaussian steps. Because this process is fixed rather than learned, it has a closed form expression at any timestep, which makes training efficient: a noisy version of an image at any point in the chain can be sampled directly, without simulating every intermediate step.

The reverse process is where learning happens. A U-Net is trained to predict the noise that was added at a given timestep, conditioned on the noisy image and the timestep itself. Training minimizes a simplified form of the evidence lower bound, derived in full in `DDPM-REPORT.pdf` (Sections 4.1 to 4.3), which reduces to a straightforward noise prediction loss.

Once trained, generating a new sample means starting from pure Gaussian noise and repeatedly applying the learned reverse step, gradually revealing a coherent image.

## Experiments and Configuration

| Model | Dataset | Timesteps | Epochs | Batch size | Learning rate | Architecture notes |
|---|---|---|---|---|---|---|
| Model 1 | MNIST | 500 | 50 | 128 | 1e-3 | Basic U-Net |
| Model 2 | MNIST | 500 | 50 | 128 | 1e-3 | Basic U-Net + self attention + transformer time embeddings + extra encoder/decoder block |
| Model 1 | CIFAR 10 | 500 | 60 | 128 | 1e-3 | MNIST model 2 architecture, adapted for RGB input |
| Model 2 | CIFAR 10 | 1000 | 60 | 128 | 1e-3 | Adds Group Normalization and additional self attention layers |
| Model 3 | CIFAR 10 | 1000 | 60 | 128 | 1e-3 | Further architectural refinement; strongest qualitative results |

All models were implemented directly in PyTorch (`torch`, `torch.nn`, `torch.nn.functional`), with `torchvision` for datasets and transforms, and `matplotlib`, `pandas`, and `seaborn` used for visualization and loss curve analysis.

## Results

**MNIST.** Model 1, using a basic U-Net, struggled to capture the underlying pixel distribution and produced noticeably lower quality samples. Model 2, with its additional encoder/decoder block, self attention, and transformer style time embeddings, produced clearer and more diverse digits. Examining training and validation loss across epochs also revealed the value of early stopping to avoid overfitting, and visualizing the reverse process at selected timesteps showed that recognizable digit structure emerges by roughly the halfway point of the denoising chain.

**CIFAR 10.** The jump from grayscale to color images proved far harder. Model 1, reusing the MNIST model 2 architecture with only an RGB input change, produced samples that were close to pure noise. Model 2, with Group Normalization and extra self attention, showed clear improvement in edge definition and brightness. Model 3 achieved the strongest alignment with real CIFAR 10 images: individual samples remain somewhat abstract, but color, edge structure, and overall coherence improved markedly across the three iterations.

**Overall conclusion.** Architectural appropriateness mattered more than raw depth. Scaling from MNIST to CIFAR 10 required not just more capacity but the right combination of normalization, attention, and a longer diffusion chain (1000 rather than 500 timesteps) before the model could reliably generate coherent color images.

## Running the Notebooks

Each notebook can be opened directly in Google Colab through the badge at the top of the file, or run locally with the following dependencies:

```
torch
torchvision
matplotlib
pandas
seaborn
```

No additional setup or external data download is required. Each notebook fetches its dataset (MNIST or CIFAR 10) directly through `torchvision.datasets` on first run.

## References

- Sohl Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. arXiv:1503.03585
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239
- Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS 30, 5998 to 6008
- Kapila, N., & collaborators. (2024). *CNNtention: Can CNNs do better with Attention?* arXiv:2412.11657
- Wang, Y., Chen, Y., Liu, X., & Zhao, L. (2024). *Development of skip connection in deep neural networks for computer vision and medical image analysis: A survey*. arXiv:2405.01725

Full derivations, proofs, and additional figures are available in `DDPM-REPORT.pdf`.
