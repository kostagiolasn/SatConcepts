# SatConcepts: Implementation of our SETN 2022 paper "Unsupervised Discovery of Semantic Concepts in Satellite Imagery with Style-based Wavelet-driven Generative Models"

## [ [paper](https://arxiv.org/abs/2208.02089) | [project page](https://kostagiolasn.github.io/setn2022/index.html) | [video](-)]

> **Unsupervised Discovery of Semantic Concepts in Satellite Imagery with Style-based Wavelet-driven Generative Models**<br>
> Nikos Kostagiolas, Mihalis A. Nicolaou, Yannis Panagakis<br>
> https://arxiv.org/abs/2208.0208 <br>
>
> **Abstract**: In recent years, considerable advancements have been made in the area of Generative Adversarial Networks (GANs), particularly with the advent of style-based architectures that address many key shortcomings - both in terms of modeling capabilities and network interpretability. Despite these improvements, the adoption of such approaches in the domain of satellite imagery is not straightforward. Typical vision datasets used in generative tasks are well-aligned and annotated, and exhibit limited variability. In contrast, satellite imagery exhibits great spatial and spectral variability, wide presence of fine, high-frequency details, while the tedious nature of annotating satellite imagery leads to annotation scarcity - further motivating developments in unsupervised learning. In this light, we present the first pre-trained style- and wavelet-based GAN model that can readily synthesize a wide gamut of realistic satellite images in a variety of settings and conditions - while also preserving high-frequency information. Furthermore, we show that by analyzing the intermediate activations of our network, one can discover a multitude of interpretable semantic directions that facilitate the guided synthesis of satellite images in terms of high-level concepts (e.g., urbanization) without using any form of supervision. Via a set of qualitative and quantitative experiments we demonstrate the efficacy of our framework, in terms of suitability for downstream tasks (e.g., data augmentation), quality of synthetic imagery, as well as generalization capabilities to unseen datasets.

## Requirements

- PyTorch 1.3.1
- CUDA/10.1.243
- GCC/8.2.0-2.31.1

### Closed-Form Factorization (https://arxiv.org/abs/2007.06600)

You can use `closed_form_factorization.py` and `apply_factor.py` to discover meaningful latent semantic factor or directions in unsupervised manner.

First, you need to extract eigenvectors of weight matrices using `closed_form_factorization.py`

> python closed_form_factorization.py [CHECKPOINT]

This will create factor file that contains eigenvectors. (Default: factor.pt) And you can use `apply_factor.py` to test the meaning of extracted directions. Use the trained SWAGAN checkpoint for the corresponding argument.

> python apply_factor.py -i [INDEX_OF_EIGENVECTOR] -d [DEGREE_OF_MOVE] -n [NUMBER_OF_SAMPLES] --ckpt [CHECKPOINT] [FACTOR_FILE]

For example,

> python apply_factor.py -i 1 -d 1 -n 50 --ckpt [CHECKPOINT] factor.pt

Will generate 50 random samples, and samples generated from latents that moved along 1st eigenvector with size/degree +-1.

![Sample of closed form factorization](doc/factor_index-1_degree-1.0.png)

## pre-trained Checkpoints

The pre-trained SWAGAN checkpoint can be found [here](https://drive.google.com/file/d/19GvThGNywddLRJoWfCEbB3VT3Sz1LZjR/view?usp=sharing). It has been trained on the RESISC-45 dataset for 200.000 iterations.

## samples

![Sample with truncation](doc/generations.png)

Sample from RESISC-45.

# citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{2208.02089,
Author = {Nikos Kostagiolas and Mihalis A. Nicolaou and Yannis Panagakis},
Title = {Unsupervised Discovery of Semantic Concepts in Satellite Imagery with Style-based Wavelet-driven Generative Models},
Year = {2022},
Eprint = {arXiv:2208.02089},
Doi = {10.1145/3549737.3549777},
}
```

## contact
**Please feel free to get in touch at**: `n.kostagiolas@cyi.ac.cy`

## credits

Our implementation contains mostly code directly from [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch), with minor alterations in order to appropriate pre-trained SWAGAN checkpoints to be loaded.
