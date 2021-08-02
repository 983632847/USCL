import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from data_aug.gaussian_blur import GaussianBlur
from data_aug.cutout import Cutout
from data_aug.outpainting import Outpainting
from data_aug.nonlin_trans import NonlinearTrans
from data_aug.sharpen import Sharpen

np.random.seed(0)


class USDataset_video(Dataset):

    def __init__(self, data_dir, transform=None, LabelList=None, DataList=None, Checkpoint_Num=None):
        """
        Ultrasound self-supervised training Dataset, choose 2 different images from a video
        :param data_dir: str
        :param transform: torch.transform
        """

        # self.label_name = {"Rb1": 0, "Rb2": 1, "Rb3": 2, "Rb4": 3, "Rb5": 4, "F0_": 5, "F1_": 6, "F2_": 7, "F3_": 8, "F4_": 9,
        #                    "Reg": 10, "Cov": 11, "Ali": 10, "Bli": 11, "Ple": 11, "Oth": 11}   # US-4
        # self.label_name = {"Rb1": 0, "Rb2": 1, "Rb3": 2, "Rb4": 3, "Rb5": 4}    # CLUST
        # self.label_name = {"F0_": 0, "F1_": 1, "F2_": 2, "F3_": 3, "F4_": 4}    # Liver Forbrosis
        self.label_name = {"Reg": 0, "Cov": 1}                                  # Butterfly
        # self.label_name = {"Ali": 0, "Bli": 1, "Ple": 2, "Oth": 3}              # COVID19-LUSMS

        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.LabelList = LabelList
        self.DataList = DataList
        self.Checkpoint_Num = Checkpoint_Num

    def __getitem__(self, index):

        # ## Different data rate
        if index not in self.DataList:
            index = random.sample(self.DataList, 1)[0] # index in data set

        path_imgs = self.data_info[index]
        if len(path_imgs) >= 3:  # more than 3 images in one video
            path_img = random.sample(path_imgs, 3)  # random choose 3 images
            img1 = Image.open(path_img[0]).convert('RGB')     # 0~255
            img2 = Image.open(path_img[1]).convert('RGB')     # 0~255
            img3 = Image.open(path_img[2]).convert('RGB')     # 0~255

            if index in self.LabelList:
                # path_imgs[0]: '/home/zhangchunhui/MedicalAI/Butte/Cov-Cardiomyopathy_mp4/Cov-Cardiomyopathy_mp4_frame0.jpg'
                # path_imgs[0][35:38]: 'Cov'
                label1 = self.label_name[path_imgs[0][35:38]]
                label2 = self.label_name[path_imgs[1][35:38]]
                label3 = self.label_name[path_imgs[2][35:38]]

            else:
                label1, label2, label3 = 9999, 9999, 9999  # unlabel data = 9999

            if self.transform is not None:
                img1, img2, img3 = self.transform((img1, img2, img3))  # transform

            ##########################################################################
            ###  frame mixup
            # alpha, beta = 2, 5
            alpha, beta = 0.5, 0.5

            lam = np.random.beta(alpha, beta)
            # img2 as anchor
            mixupimg1 = lam * img1 + (1.0 - lam) * img2
            mixupimg2 = lam * img3 + (1.0 - lam) * img2

            return mixupimg1, label1, mixupimg2, label2, img1, img2

        elif len(path_imgs) == 2:
            path_img = random.sample(path_imgs, 2)  # random choose 3 images
            img1 = Image.open(path_img[0]).convert('RGB')     # 0~255
            img2 = Image.open(path_img[1]).convert('RGB')     # 0~255
            if index in self.LabelList:
                label1 = self.label_name[path_imgs[0][35:38]]
                label2 = self.label_name[path_imgs[1][35:38]]
            else:
                label1, label2 = 9999, 9999     # unlabel data = 9999

            if self.transform is not None:
                img1, img2 = self.transform((img1, img2))  # transform

            return img1, label1, img2, label2, img1, img2

        else:  # one image in the video, using augmentation to obtain two positive samples
            img1 = Image.open(path_imgs[0]).convert('RGB')  # 0~255
            img2 = Image.open(path_imgs[0]).convert('RGB')  # 0~255
            if index in self.LabelList:
                label1 = self.label_name[path_imgs[0][35:38]]
                label2 = self.label_name[path_imgs[0][35:38]]
            else:
                label1, label2 = 9999, 9999  # unlabel data = 9999

            if self.transform is not None:
                img1, img2 = self.transform((img1, img2))  # transform

            return img1, label1, img2, label2, img1, img2

        # if self.transform is not None:
        #     img1, img2 = self.transform((img1, img2))  # transform
        # return img1, label1, img2, label2

    def __len__(self):  # len
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:  # one video as one class
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))

                path_imgs = []  # list
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    path_imgs.append(path_img)
                data_info.append(path_imgs)

        return data_info


class USDataset_image(Dataset):

    def __init__(self, data_dir, transform=None, LabelList=None, DataList=None):
        """
        Ultrasound self-supervised training Dataset, only choose one image from a video
        :param data_dir: str
        :param transform: torch.transform
        """
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.LabelList = LabelList
        self.DataList = DataList

    def __getitem__(self, index):
        path_imgs = self.data_info[index]  # list
        path_img = random.sample(path_imgs, 1)  # random choose one image
        img1 = Image.open(path_img[0]).convert('RGB')  # 0~255
        img2 = Image.open(path_img[0]).convert('RGB')  # 0~255
        label1 = 0 if path_img[0].lower()[64:].find("cov") > -1 else (1 if path_img[0].lower()[64:].find("pneu") > -1 else 2)

        if self.transform is not None:
            img1, img2 = self.transform((img1, img2))  # transform

        return img1, label1, img2, label1

    def __len__(self):  # len
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:  # one video as one class
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.png'), img_names))
                path_imgs = []
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    path_imgs.append(path_img)
                data_info.append(path_imgs)

        return data_info


class DataSetWrapper(object):

    def __init__(self, batch_size, LabelList, DataList, Checkpoint_Num, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size  # leave out ratio, e.g. 0.20
        self.s = s
        self.input_shape = eval(input_shape)  # (H, W, C) shape of input image
        self.LabelList = LabelList
        self.DataList = DataList
        self.Checkpoint_Num = Checkpoint_Num

    def get_data_loaders(self):
        ''' Get dataloader for target dataset, this function will be called before the training process '''

        data_augment = self._get_simclr_pipeline_transform()
        print('\nData augmentation:')
        print(data_augment)

        use_video = True
        if use_video:
            print('\nUse video augmentation!')
            # US-4
            # train_dataset = USDataset_video("/home/zhangchunhui/WorkSpace/SSL/Ultrasound_Datasets_train/Video/",
            #                                 transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 2 images

            # 1 video-CLUST
            # train_dataset = USDataset_video("/home/zhangchunhui/MedicalAI/Ultrasound_Datasets_train/CLUST/",
            #                                 transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 2 images
            # 1 video-Liver
            # train_dataset = USDataset_video("/home/zhangchunhui/MedicalAI/Ultrasound_Datasets_train/Liver/",
            #                                 transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 2 images
            # 1 video-COVID
            # train_dataset = USDataset_video("/home/zhangchunhui/MedicalAI/Ultrasound_Datasets_train/COVID/",
            #                                 transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 2 images
            # 1 video-Butte
            train_dataset = USDataset_video("/home/zhangchunhui/MedicalAI/Butte/",
                                            transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 2 images
        else:
            print('\nDo not use video augmentation!')
            # Images
            train_dataset = USDataset_image("/home/zhangchunhui/MedicalAI/Butte/",
                                            transform=SimCLRDataTransform(data_augment), LabelList=self.LabelList, DataList=self.DataList)  # augmented from 1 image

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        # train_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader
        # return train_loader

    def __len__(self): #
        return self.batch_size

    def _get_simclr_pipeline_transform(self):
        '''
        Get a set of data augmentation transformations as described in the SimCLR paper.
        Random Crop (resize to original size) + Random color distortion + Gaussian Blur
        '''

        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([Sharpen(degree=0),
                                             transforms.Resize((self.input_shape[0], self.input_shape[1])),
                                             transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0), ratio=(0.8, 1.25)),
                                             transforms.RandomHorizontalFlip(),
                                             # transforms.RandomRotation(10),
                                             color_jitter,
                                             # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                                             # GaussianBlur(kernel_size=int(0.05 * self.input_shape[0])),
                                             transforms.ToTensor(),
                                             # NonlinearTrans(prob=0.9),   # 0-1
                                             # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
                                             # Cutout(n_holes=3, length=32),
                                             # Outpainting(n_holes=5),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
                                              ])

        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain indices that will be used for training / validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_idx= indices[split:]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # data loaders for training and validation, drop_last should be False to avoid data shortage of valid_loader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader
        # return train_loader



class SimCLRDataTransform(object):
    ''' transform two images in a video to two augmented samples '''

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        if len(sample)>2:
            xi = self.transform(sample[0])  # sample -> xi, xj in original implementation
            xj = self.transform(sample[1])
            xk = self.transform(sample[2])
            return xi, xj, xk
        else:
            xi = self.transform(sample[0])  # sample -> xi, xj in original implementation
            xj = self.transform(sample[1])
            return xi, xj
