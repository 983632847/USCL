# USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning

### Accepted by MICCAI 2021 (Oral). [Paper Link](https://arxiv.org/abs/2011.13066)

This repository includes the constructed **US-4** dataset and source codes (PyTorch version) of USCL, to appear in MICCAI 2021.

### Abstract
Most deep neural networks (DNNs) based ultrasound (US) medical image analysis models use pretrained backbones (e.g., ImageNet) for better model generalization. However, the domain gap between natural and medical images causes an inevitable performance bottleneck when applying to US image analysis. Our idea is to pretrain DNNs on US images directly to avoid this bottleneck. Due to the lack of annotated large-scale datasets of US images, we first construct a new large-scale US video-based image dataset named US-4, containing over 23,000 high resolution images from four US video sub-datasets, where two sub-datasets are newly collected by our local experienced doctors. To make full use of this dataset, we then innovatively propose an US semi-supervised contrastive learning (USCL) method to effectively learn feature representations of US images, with a new sample pair generation (SPG) scheme to tackle the problem that US images extracted from videos have high similarities. Moreover, the USCL treats contrastive loss as a consistent regularization, which boosts the performance of pretrained backbones by combining the supervised loss in a mutually reinforcing way. Extensive experiments on down-stream tasks' fine-tuning show the superiority of our approach against ImageNet pretraining and pretraining using previous state-of-the-art semi-supervised learning approaches. In particular, our pretrained backbone gets fine-tuning accuracy of over 94%, which is 9% higher than 85% of the ImageNet pretrained model on the widely used POCUS dataset. The constructed US-4 dataset and source codes of this work will be made public.

![image](https://github.com/983632847/USCL/blob/main/framework.png)


### US-4 dataset
The US-4 dataset will be available at [US-4](https://github.com/983632847/USCL).

![image](https://github.com/983632847/USCL/blob/main/Examples_US4.png)


### Quick Start

#### Fine-tune with Pretrained Model
1. Pick a model and its config file, for example, `config.yaml`
2. Download the model `best_model.pth`
3. Download the 5 fold cross validation [POCUS](https://drive.google.com/file/d/111lHpStoY_gYMhCQ-Yt95AreDx0G7-2R/view?usp=sharing) dataset
4. Run the demo with
```
python eval_pretrained_model.py
```

Our pretrained ResNet-18 model on US-4 dataset can be downloaded as following:

Name | epochs | Project head | Classifier | Accuracy | download
---  |:---------:|:---------:|:---------:|:---------:|:---:
ResNet-18 | 300 | Yes | Yes | 94.19 | [model](https://drive.google.com/file/d/1ODH2oeZxZdblmEW725AuZYA51AT9QJH2/view?usp=sharing)


This repository reports the fine-tuning accuracy (%) on [POCUS](https://arxiv.org/abs/2004.12084) dataset.


#### Train Your Own Model
1. Download the [Butterfly](https://pan.baidu.com/s/1tQtDzoditkTft3LMeDfGqw) (Pwd:butt) dataset
2. Train the USCL model with
```
python run.py
```


#### Environment
The code is developed with an Intel Xeon Silver 4210R CPU @ 2.4GHz and a single Nvidia Tesla V100 GPU.

The install script has been tested on an Ubuntu 18.04 system.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

### License

Licensed under an MIT license.

### Citation

If you find the code and dataset useful in your research, please consider citing:

    @article{Chen2021MICCAI,
        title={USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning},
        author = {Yixiong Chen, and Chunhui Zhang, and Li Liu, and Cheng Feng, and Changfeng Dong, and Yongfang Luo, and Xiang Wan},
        journal = {MICCAI},
        year = {2021}
      }


     @article{born2021accelerating,
        title={Accelerating detection of lung pathologies with explainable ultrasound image analysis},
        author={Born, Jannis and Wiedemann, Nina and Cossio, Manuel and Buhre, Charlotte and Br{\"a}ndle, Gabriel and Leidermann, Konstantin and Aujayeb, Avinash and Moor, Michael             and Rieck, Bastian and Borgwardt, Karsten},
        journal={Applied Sciences},
        pages={672},
        year={2021},
        }


     @inproceedings{Somphone2014MICCAIW,
        author={O. Somphone, and S. Allaire, and B. Mory, and C. Dufour},
        title={Live Feature Tracking in Ultrasound Liver Sequences with Sparse Demons},
        booktitle = {International Conference on Medical Image Computing and Computer Assisted Intervention Workshop},
        pages = {53-60}, 
        year={2014},
      }


### Contact
Feedbacks and comments are welcome! Feel free to contact us via [andyzhangchunhui@gmail.com](mailto:andyzhangchunhui@gmail.com) or [16307110231@fudan.edu.cn](mailto:16307110231@fudan.edu.cn) or [liuli@cuhk.edu.cn](mailto:liuli@cuhk.edu.cn).

Enjoy!
