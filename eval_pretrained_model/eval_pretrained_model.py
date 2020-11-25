import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from tools.my_dataset import COVIDDataset
from resnet_uscl import ResNetUSCL

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    print("Apex on, run on mixed precision.")
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nRunning on:", device)

if device == 'cuda':
    device_name = torch.cuda.get_device_name()
    print("The device name is:", device_name)
    cap = torch.cuda.get_device_capability(device=None)
    print("The capability of this device is:", cap, '\n')

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    # ============================ step 1/5 data ============================
    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    # MyDataset
    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    valid_data = COVIDDataset(data_dir=data_dir, train=False, transform=valid_transform)

    # DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 model ============================

    net = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=pretrained)
    if pretrained:
        print('\nThe ImageNet pretrained parameters are loaded.')
    else:
        print('\nThe ImageNet pretrained parameters are not loaded.')

    if selfsup: # import pretrained model weights
        state_dict = torch.load(state_dict_path)
        new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                    if not (k.startswith('l')
                            | k.startswith('fc'))}  # # discard MLP and fc
        model_dict = net.state_dict()

        model_dict.update(new_dict)
        net.load_state_dict(model_dict)
        print('\nThe self-supervised trained parameters are loaded.\n')
    else:
        print('\nThe self-supervised trained parameters are not loaded.\n')

    # frozen all convolutional layers
    # for param in net.parameters():
    #     param.requires_grad = False

    # fine-tune last 3 layers
    for name, param in net.named_parameters():
        if not name.startswith('features.7.1'):
            param.requires_grad = False

    # add a classifier for linear evaluation
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, 3)
    net.fc = nn.Linear(3, 3)

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)

    net.to(device)

    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()       # choose loss function

    # ============================ step 4/5 optimizer ============================
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)      # choose optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=MAX_EPOCH, 
                                                     eta_min=0,
                                                     last_epoch=-1)     # set learning rate decay strategy

    # ============================ step 5/5 training ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0    # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None

    if apex_support and fp16_precision:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)
    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            if apex_support and fp16_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update weights
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # print training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        print('Learning rate this epoch:', scheduler.get_last_lr()[0])
        scheduler.step()  # updata learning rate

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1    # "label" is regarded as "predicted"

                    loss_val += loss.item()

                acc = correct_val / total_val
                if acc > max_acc:   # record best accuracy
                    max_acc = acc
                    reached = epoch
                    best_classification_results = classification_results
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, acc))

    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))

    print('\nThe best prediction results of the dataset:')
    print('Class 0 predicted as class 0:', best_classification_results[0][0])
    print('Class 0 predicted as class 1:', best_classification_results[0][1])
    print('Class 0 predicted as class 2:', best_classification_results[0][2])
    print('Class 1 predicted as class 0:', best_classification_results[1][0])
    print('Class 1 predicted as class 1:', best_classification_results[1][1])
    print('Class 1 predicted as class 2:', best_classification_results[1][2])
    print('Class 2 predicted as class 0:', best_classification_results[2][0])
    print('Class 2 predicted as class 1:', best_classification_results[2][1])
    print('Class 2 predicted as class 2:', best_classification_results[2][2])

    acc0 = best_classification_results[0][0] / sum(best_classification_results[i][0] for i in range(3))
    recall0 = best_classification_results[0][0] / sum(best_classification_results[0])
    print('\nClass 0 accuracy:', acc0)
    print('Class 0 recall:', recall0)
    print('Class 0 F1:', 2 * acc0 * recall0 / (acc0 + recall0))

    acc1 = best_classification_results[1][1] / sum(best_classification_results[i][1] for i in range(3))
    recall1 = best_classification_results[1][1] / sum(best_classification_results[1])
    print('\nClass 1 accuracy:', acc1)
    print('Class 1 recall:', recall1)
    print('Class 1 F1:', 2 * acc1 * recall1 / (acc1 + recall1))

    acc2 = best_classification_results[2][2] / sum(best_classification_results[i][2] for i in range(3))
    recall2 = best_classification_results[2][2] / sum(best_classification_results[2])
    print('\nClass 2 accuracy:', acc2)
    print('Class 2 recall:', recall2)
    print('Class 2 F1:', 2 * acc2 * recall2 / (acc2 + recall2))
    
    return best_classification_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-p', '--path', default='checkpoint', help='folder of ckpt')
    args = parser.parse_args()

    set_seed(1)  # random seed

    # parameters
    MAX_EPOCH = 100       # default = 100
    BATCH_SIZE = 32       # default = 32
    LR = 0.01             # default = 0.01
    weight_decay = 1e-4   # default = 1e-4
    log_interval = 10
    val_interval = 1
    base_path = "./eval_pretrained_model/"
    state_dict_path = os.path.join(base_path, args.path, "best_model.pth")
    print('State dict path:', state_dict_path)
    fp16_precision = True
    pretrained = False
    selfsup = True

    # save result
    save_dir = os.path.join('result')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    resultfile = save_dir + '/my_result.txt'

    if (not (os.path.exists(resultfile))) and (os.path.exists(state_dict_path)):
        confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(1, 6):
            print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
            data_dir = "./covid_5_fold/covid_data{}.pkl".format(i)
            best_classification_results = main()
            confusion_matrix = confusion_matrix + np.array(best_classification_results)

        print('\nThe confusion matrix is:')
        print(confusion_matrix)
        print('\nThe precision of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[:,0]))
        print('The precision of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[:,1]))
        print('The precision of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[:,2]))
        print('\nThe recall of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[0]))
        print('The recall of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[1]))
        print('The recall of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[2]))
        print('\nTotal acc is:', (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum())

        file_handle = open(save_dir + '/my_result.txt', mode='w+')
        file_handle.write("precision 0: "+str(confusion_matrix[0,0] / sum(confusion_matrix[:,0]))); file_handle.write('\r\n')
        file_handle.write("precision 1: "+str(confusion_matrix[1,1] / sum(confusion_matrix[:,1]))); file_handle.write('\r\n')
        file_handle.write("precision 2: "+str(confusion_matrix[2,2] / sum(confusion_matrix[:,2]))); file_handle.write('\r\n')
        file_handle.write("recall 0: "+str(confusion_matrix[0,0] / sum(confusion_matrix[0]))); file_handle.write('\r\n')
        file_handle.write("recall 1: "+str(confusion_matrix[1,1] / sum(confusion_matrix[1]))); file_handle.write('\r\n')
        file_handle.write("recall 2: "+str(confusion_matrix[2,2] / sum(confusion_matrix[2]))); file_handle.write('\r\n')
        file_handle.write("Total acc: "+str((confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum()))

        file_handle.close()