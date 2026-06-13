# Neural Network Hardware Format Evaluator

This repository contains standalone PyTorch scripts for training and inference of deep learning models (ConvNeXt, ResNeXt, VGG-16) on the CIFAR-10 dataset. 

The core purpose of this project is to evaluate the impact of various native numerical formats and precision constraints on model performance, simulating hardware-level datapaths. 

## Supported Numerical Formats
The custom `apply_format` function and pure-format autograd classes support:
* **Floating Point:** `FP32`, `FP16`
* **Fixed Point:** `FXP32` (Q16.16), `FXP16` (Q8.8)
* **Posit (SoftPosit):** `POSIT32_2`, `POSIT16_1`, `POSIT8_0`

## Architectures
The standard layers are replaced with custom `HardwareConv2d`, `HardwareLinear`, and `HardwareLayerNormChannelsFirst` modules to enforce strict mathematical format boundaries during both forward passes and backpropagation.
* **VGG-16**
* **ResNeXt** (2x64d equivalent)
* **ConvNeXt**

## Setup and Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt