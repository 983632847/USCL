# Effective Sample Pair Generation for Ultrasound Video Contrastive Representation Learning

### Introduction

This repository includes the constructed **US-4** dataset and source codes for **USCL** (PyTorch version).

### Abstract
Most deep neural networks (DNNs) based ultrasound (US) medical image analysis models use pretrained backbones (e.g., ImageNet) for better model generalization. However, the domain gap between natural and medical images causes an inevitable performance bottleneck when applying to US image analysis. Our idea is to pretrain DNNs on US images directly to avoid this bottleneck. Due to the lack of annotated large-scale datasets of US images, we first construct a new large-scale US video-based image dataset named US-4, containing over 23,000 high resolution images from four US video sub-datasets, where two sub-datasets are newly collected by our local experienced doctors. To make full use of this dataset, we then innovatively propose an US semi-supervised contrastive learning (USCL) method to effectively learn feature representations of US images, with a new sample pair generation (SPG) scheme to tackle the problem that US images extracted from videos have high similarities. Moreover, the USCL treats contrastive loss as a consistent regularization, which boosts the performance of pretrained backbones by combining the supervised loss in a mutually reinforcing way. Extensive experiments on down-stream tasks' fine-tuning show the superiority of our approach against ImageNet pretraining and pretraining using previous state-of-the-art semi-supervised learning approaches. In particular, our pretrained backbone gets fine-tuning accuracy of over 94%, which is 9% higher than 85% of the ImageNet pretrained model on the widely used POCUS dataset. The constructed US-4 dataset and source codes of this work will be made public.

![image](https://github.com/983632847/USCL/blob/main/framework.png)


### US-4 dataset
The US-4 dataset will be available at [US-4](https://github.com/983632847/USCL).

![image](https://github.com/983632847/USCL/blob/main/Examples_US4.png)


### Quick Start

#### Fine-tune with Pretrained Model
Coming soon.


#### Train Your Own Model
Coming soon.


Our pretrained ResNet-18 model on US-4 dataset can be downloaded as following:

Name | epochs | Project head | Classifier | Accuracy | download
---  |:---------:|:---------:|:---------:|:---------:|:---:
ResNet-18 | 300 | Yes | Yes | 94.19 | [model](https://drive.google.com/file/d/1ODH2oeZxZdblmEW725AuZYA51AT9QJH2/view?usp=sharing)


This repository reports fine-tuning accuracy (%) on [POCUS](https://arxiv.org/abs/2004.12084) dataset.


#### Environment
The code is developed with an Intel Xeon Silver 4210R CPU @ 2.4GHz and a single Nvidia Tesla V100 GPU.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

### License

Licensed under an MIT license.

### Citation

If you find the code and dataset useful in your research, please consider citing:

    @article{USCL2020arXiv,
        title={Effective Sample Pair Generation for Ultrasound Video Contrastive Representation Learning},
        author = {Yixiong Chen, and Chunhui Zhang, and Li Liu, and Cheng Feng, and Changfeng Dong, and Yongfang Luo, and Xiang Wan},
        journal = {arXiv},
        year = {2020}
      }


    @article{born2020arXiv,
        title={POCOVID-Net: automatic detection of COVID-19 from a new lung ultrasound imaging dataset (POCUS)},
        author={Born Jannis, and Br{\"a}ndle, Gabriel, and Cossio Manuel, and Disdier Marion, and Goulet Julie, and Roulin J{\'e}r{\'e}mie, and Wiedemann Nina},
        journal={arXiv:2004.12084},
        year={2020}
      }


    @inproceedings{Somphone2014MICCAIW,
        author={O. Somphone, and S. Allaire, and B. Mory, and C. Dufour},
        title={Live Feature Tracking in Ultrasound Liver Sequences with Sparse Demons},
        booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention Workshop},
        pages = {53-60}, 
        year={2014},
      }


### Contact
Feedbacks and comments are welcome! Feel free to contact us via [zhangchunhui@iie.ac.cn](zhangchunhui@iie.ac.cn) or [16307110231@fudan.edu.cn](mailto:16307110231@fudan.edu.cn) or [liuli@cuhk.edu.cn](mailto:liuli@cuhk.edu.cn).

Enjoy!
