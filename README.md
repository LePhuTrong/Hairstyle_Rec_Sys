## Prerequisites
You need following hardware and python version to run our method.
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.10
- PyTorch 1.13.1+
- Git LFS
- Anaconda (recommended)
- gdown

## Installation

* Clone this repo:
```bash
git clone https://github.com/LePhuTrong/Hairstyle_Rec_Sys.git Hairstyle_Rec
cd Hairstyle_Rec
```

* Download all pretrained models:
(At Hairstyle_Rec folder)
```bash
git clone https://huggingface.co/AIRI-Institute/HairFastGAN
cd HairFastGAN && git lfs pull && cd ..
mv HairFastGAN/pretrained_models pretrained_models
mv HairFastGAN/input input
rm -rf HairFastGAN

gdown --id 1dNrNE1PEld57bCBLIGBvcPAv_RMWZgy1 -O ./onnx_models/swinface.onnx #swinface.onnx
```

* Setting the environment
    - Creating new virtual environment is recommended!
        ```bash
        pip install -r requirements.txt
        ```

## Run the project
- All dependencies of project is included in demo_new.ipynb notebook!