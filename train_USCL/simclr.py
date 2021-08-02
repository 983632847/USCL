import torch
from models.resnet_simclr import ResNetSimCLR
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
import time
import torch.nn as nn

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    print("Apex on, run on mixed precision.")

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(2)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

def FindNotX(list, x):
    index = []
    for i, item in enumerate(list):
        if item != x:
            index.append(i)
    return index

class SimCLR(object):

    def __init__(self, dataset, config, lumbda, Checkpoint_Num):
        self.lumbda1 = lumbda
        self.lumbda2 = lumbda
        self.config = config
        self.Checkpoint_Num = Checkpoint_Num
        print('\nThe configurations of this model are in the following:\n', config)
        self.device = self._get_device()
        # self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\nRunning on:", device)

        if device == 'cuda':
            device_name = torch.cuda.get_device_name()
            print("The device name is:", device_name)
            cap = torch.cuda.get_device_capability(device=None)
            print("The capability of this device is:", cap, '\n')

        return device

    def _step(self, model, xis, xjs):

        # get the representations and the projections
        ris, zis, labelis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs, labeljs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        criterion = nn.CrossEntropyLoss()  # loss function

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'], eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(
            '/home/zhangchunhui/MedicalAI/USCL/checkpoints_multi_aug',
            'checkpoint_' + str(self.Checkpoint_Num))

        # save config file
        _save_config_file(model_checkpoints_folder)

        start_time = time.time()
        end_time = time.time()
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch in range(self.config['epochs']):
            for i, data in enumerate(train_loader, 1):
                # forward
                # mixupimg1, label1, mixupimg2, label2, original img1, original img2
                xis, labelis, xjs, labeljs, imgis, imgjs = data  # N samples of left branch, N samples of right branch

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)


                ####### 1-Semi-supervised
                hi, xi, outputis = model(xis)
                hj, xj, outputjs = model(xjs)
                labelindexi, labelindexj = FindNotX(labelis.tolist(), 9999), FindNotX(labeljs.tolist(), 9999) # X=9999=no label

                lossi = criterion(outputis[labelindexi], labelis.to(self.device)[labelindexi])
                lossj = criterion(outputjs[labelindexj], labeljs.to(self.device)[labelindexj])

                # lumbda1=lumbda2   # small value is better
                lumbda1, lumbda2 = self.lumbda1, self.lumbda2  # small value is better
                loss = self._step(model, xis, xjs) + lumbda1 * lossi + lumbda2 * lossj
                ########################################################################################################


                ####### 2-Self-supervised
                # loss = self._step(model, xis, xjs)
                ########################################################################################################

                # backward
                optimizer.zero_grad()
                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # update weights
                optimizer.step()

                if i % self.config['log_every_n_steps'] == 0:
                    # self.writer.add_scalar('train_loss', loss, global_step=i)
                    start_time, end_time = end_time, time.time()
                    print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s".format(
                        epoch + 1, self.config['epochs'], i, len(train_loader), loss, end_time - start_time))

            # validate the model if requested
            if epoch % self.config['eval_every_n_epochs'] == 0:
                start_time = time.time()
                valid_loss = self._validate(model, valid_loader)
                end_time = time.time()
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'best_model.pth'))

                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Time: {:.2f}s".format(
                    epoch + 1, self.config['epochs'], len(valid_loader), len(valid_loader), valid_loss,
                    end_time - start_time))
                # self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            print('Learning rate this epoch:', scheduler.get_last_lr()[0])  # python >=3.7
            # print('Learning rate this epoch:', scheduler.base_lrs[0])   # python 3.6

            # warmup for the first 10 epochs
            if epoch >= 10:
                scheduler.step()
            # self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=i)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for xis, labelis, xjs, labeljs, imgis, imgjs in valid_loader:
                ## 1. original images
                xis = imgis.to(self.device)
                xjs = imgjs.to(self.device)

                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1

                ## 2. augmented images
                # xis = xis.to(self.device)
                # xjs = xjs.to(self.device)
                #
                # loss = self._step(model, xis, xjs)
                # valid_loss += loss.item()
                # counter += 1
            valid_loss /= (counter + 1e-6)  # in case 0
        model.train()
        return valid_loss
