# StyleGAN 2 in PyTorch

Implementation of our SETN paper, entitled: "Unsupervised Discovery of Semantic Concepts in Satellite Imagery with Style-based Wavelet-driven Generative Models".

## Notice

This implementation is based on the rosalinity StyleGAN2 one (SWAGAN included), which can be found [here](https://github.com/rosinality/stylegan2-pytorch).

## Requirements

- PyTorch 1.3.1
- Anaconda3
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

![Sample of closed form factorization](factor_index-1_degree-1.0.png)

## Pretrained Checkpoints

The pre-trained SWAGAN checkpoint can be found in Google Drive. It has been trained on the RESISC-45 dataset for 200.000 iterations.

## Samples

![Sample with truncation](doc/generations.png)

Sample from RESISC-45.

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
