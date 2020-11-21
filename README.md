# Effective Sample Pair Generation for Ultrasound Video Contrastive Representation Learning

### Introduction

This repository includes constructed **US-4** dataset and the code for **USCL** (PyTorch version).

### Abstract
Most deep neural networks (DNNs) based ultrasound (US) medical image analysis models use pretrained backbones (\eg., ImageNet) for better model generalization. However, the domain gap between natural and medical images causes an inevitable performance bottleneck when applying to US image analysis. Our idea is to pretrain DNNs on US images directly to avoid this bottleneck. Due to the lack of annotated large-scale datasets of US images, we first construct a new large-scale US video-based image dataset named US-4, containing over 23,000 high resolution images from four US video sub-datasets, where two sub-datasets are newly collected by our local experienced doctors. To make full use of this dataset, we then innovatively propose an US semi-supervised contrastive learning (USCL) method to effectively learn feature representations of US images, with a new sample pair generation (SPG) scheme to tackle the problem that US images extracted from videos have high similarities. Moreover, the USCL treats contrastive loss as a consistent regularization, which boosts the performance of pretrained backbones by combining the supervised loss in a mutually reinforcing way. Extensive experiments on down-stream tasks' fine-tuning show the superiority of our approach against ImageNet pretraining and pretraining using previous state-of-the-art semi-supervised learning approaches. In particular, our pretrained backbone gets fine-tuning accuracy of over 94%, which is 9% higher than 85% of the ImageNet pretrained model on the widely used POCUS dataset. The constructed US-4 dataset and source codes of this work will be made public.

![image](https://github.com/983632847/USCL/framework.png)


### US-4 dataset


### Demo 


### Pretraing code 


### Fine-tune code 


### Contact
Feedbacks and comments are welcome! Feel free to contact us via [zhangchunhui@iie.ac.cn](zhangchunhui@iie.ac.cn) or [16307110231@fudan.edu.cn](mailto:16307110231@fudan.edu.cn) or [liuli@cuhk.edu.cn](mailto:liuli@cuhk.edu.cn).

Enjoy!
