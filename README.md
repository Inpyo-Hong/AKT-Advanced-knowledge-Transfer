<div align="center">

# Advanced Knowledge Transfer: Refined Feature Distillation for Zero-Shot Quantization in Edge Computing

**Inpyo Hong**, **Youngwan Jo**, **Hyojeong Lee**, **Sunghyun Ahn**, and **Sanghyun Park**

[![ACM SAC](https://img.shields.io/badge/ACM%20SAC-2025-005596)](https://dl.acm.org/doi/abs/10.1145/3672608.3707816)

</div>


## Abstract
We introduce AKT (Advanced Knowledge Transfer), a novel method to enhance the training ability of low-bit quantized (Q) models in the field of zero-shot quantization (ZSQ). Existing research in ZSQ has focused on generating high-quality data from full-precision (FP) models. However, these approaches struggle with reduced learning ability in low-bit quantization due to its limited information capacity. To overcome this limitation, we propose an effective training strategy compared to data generation. Particularly, we analyzed that refining feature maps in the feature distillation process is an effective way to transfer knowledge to the Q model. Based on this analysis, AKT efficiently transfers core information from the FP model to the Q model. AKT is the first approach to utilize both spatial and channel attention information in feature distillation in ZSQ. Our method addresses the fundamental gradient exploding problem in low-bit Q models. Experiments on CIFAR-10 and CIFAR-100 datasets demonstrated the effectiveness of the AKT. Our method led to significant performance enhancement in existing generative models. Notably, AKT achieved significant accuracy improvements in low-bit Q models, achieving state-of-the-art in the 3-bit and 5-bit scenarios on CIFAR-10. The code is available at https://github.com/Inpyo-Hong/AKT-Advanced-knowledge-Transfer.

![Fig1](https://github.com/user-attachments/assets/9c5882e2-e983-4597-9ea1-48c3956fda29)


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
