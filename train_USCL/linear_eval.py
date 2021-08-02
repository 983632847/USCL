import os
import yaml
import pickle
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import importlib.util

##################################### 设定 ####################################
fold = 1
self_supervised_pretrained = True
model_path = '/home/zhangchunhui/WorkSpace/SSL/checkpoints_multi_aug/checkpoint_resnet18/best_model.pth'
out_dim = 256
base_model = 'resnet18'
pretrained = False # 初始化模型时，是否载入ImageNet预训练参数
batch_size = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

checkpoints_folder = '/home/zhangchunhui/WorkSpace/SSL/checkpoints_multi_aug/checkpoint_resnet18/'
config = yaml.load(open(os.path.join(checkpoints_folder, "config.yaml"), "r"), Loader=yaml.Loader)
print("\nConfig:\n", config)

###############################################################################

# load covid ultrasound data
with open('/home/zhangchunhui/WorkSpace/SSL/covid_5_fold/covid_data{}.pkl'.format(fold), 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

def linear_model_eval(X_train, y_train, X_test, y_test):
    
    clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
    clf.fit(X_train, y_train)
    print("\nLogistic Regression feature eval")
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))
    
    print("-------------------------------")
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(X_train, y_train)
    print("KNN feature eval")
    print("Train score:", neigh.score(X_train, y_train))
    print("Test score:", neigh.score(X_test, y_test))

def next_batch(X, y, batch_size, dtype):
    for i in range(0, X.shape[0], batch_size):
        # must convert data type to type of weights
        X_batch = torch.tensor(X[i: i+batch_size], dtype=dtype) / 255.
        y_batch = torch.tensor(y[i: i+batch_size])
        yield X_batch.to(device), y_batch.to(device)

################################ 定义模型与参数 #################################

# Load the neural net module
spec = importlib.util.spec_from_file_location("model", '/models/resnet_simclr_copy.py')
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)

model = resnet_module.ResNetSimCLR(base_model, out_dim, pretrained)
model.eval()

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
weight_dtype = state_dict[list(state_dict.keys())[0]].dtype
if self_supervised_pretrained:
    print('Load self-supervised model parameters.')
    model.load_state_dict(state_dict)
model = model.to(device)

################################ 训练集的特征 ##################################

X_train_feature = []

for batch_x, batch_y in next_batch(X_train, y_train, batch_size=batch_size, dtype=weight_dtype):
    features, _ = model(batch_x) # 只要输出的特征向量，不要投影头输出的Logit
    X_train_feature.extend(features.cpu().detach().numpy())
    
X_train_feature = np.array(X_train_feature)

print("Train features")
print(X_train_feature.shape)

################################ 测试集的特征 ##################################

X_test_feature = []

for batch_x, batch_y in next_batch(X_test, y_test, batch_size=batch_size, dtype=weight_dtype):
    features, _ = model(batch_x)
    X_test_feature.extend(features.cpu().detach().numpy())
    
X_test_feature = np.array(X_test_feature)

print("Test features")
print(X_test_feature.shape)

################################## 评估特征 ###################################

scaler = preprocessing.StandardScaler()
scaler.fit(X_train_feature) # 获取用于特征标准化的均值方差

linear_model_eval(scaler.transform(X_train_feature), y_train, scaler.transform(X_test_feature), y_test)

del X_train_feature
del X_test_feature

################################# 计算总准确率 #################################

def cal_total_acc(acc_list):
    acc = (acc_list[0]*476 + acc_list[1]*369 + acc_list[2]*487 + acc_list[3]*360 + acc_list[4]*424)/2116
    return acc

































