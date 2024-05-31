## Prerequisites
You need following hardware and python version to run our method.
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.10
- PyTorch 1.13.1+

## Installation

* Clone this repo:
```bash
git clone https://github.com/AIRI-Institute/HairFastGAN
cd HairFastGAN
```

* Download all pretrained models:
```bash
git clone https://huggingface.co/AIRI-Institute/HairFastGAN
cd HairFastGAN && git lfs pull && cd ..
mv HairFastGAN/pretrained_models pretrained_models
mv HairFastGAN/input input
rm -rf HairFastGAN
```