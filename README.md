# Advanced Knowledge Transfer: Refined Feature Distillation for Zero-Shot Quantization in Edge Computing
This repository is the official code for the paper "Advanced Knowledge Transfer: Refined Feature Distillation for Zero-Shot Quantization in Edge Computing" by Inpyo Hong, Youngwan Jo, Hyojeong Lee, Sunghyun Ahn, and Sanghyun Park.

## Abstract
We introduce AKT (Advanced Knowledge Transfer), a novel method to enhance the training ability of low-bit quantized (Q) models in the field of zero-shot quantization (ZSQ). Existing research in ZSQ has focused on generating high-quality data from full-precision (FP) models . However, these approaches struggle with reduced learning ability in low-bit quantization due to its limited information capacity. To overcome this limitation, we propose effective training strategy compared to data generation. Particularly, we analyzed that refining feature maps in the feature distillation process is an effective way to transfer knowledge to the Q model. Based on this analysis, AKT efficiently transfer core information from the FP model to the Q model. AKT is the first approach to utilize both spatial and channel attention information in feature distillation in ZSQ. Our method addresses the fundamental gradient exploding problem in low-bit Q models. Experiments on CIFAR-10 and CIFAR-100 datasets demonstrated the effectiveness of the AKT. Our method led to significant performance enhancement in existing generative models. Notably, AKT achieved significant accuracy improvements in low-bit Q models, achieving state-of-the-art in the 3,5bit scenarios on CIFAR-10.

![Fig1](https://github.com/user-attachments/assets/f5041b8e-7be3-418d-a164-c2e512d9d58c)


## Requirements
python == 3.6 <br/>
pytorch == 1.3.1

## Usage
Example of 4-bit Quantization on AKT.
```
bash run_4w4a.sh
```

## Experimental Results
Top-1 accuracy comparison of AKT with zero-shot quantization methods.
![Table2](https://github.com/user-attachments/assets/a2d64461-8bdf-4908-b115-0bac36de9353)
