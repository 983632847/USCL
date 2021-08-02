# -*- coding:utf-8 -*-
'''
Created on 2017年10月28日

@summary: 利用Python实现NMI计算

@author: dreamhome
'''
import math
import numpy as np
from sklearn import metrics
import time
import random
import torch

def MILoss(TensorA=None, TensorB=None):
    # TensorA, TensorB = range(112*512*7*7), range(112*512*7*7)
    #
    # TensorA, TensorB = np.array(TensorA), np.array(TensorB)
    # TensorA, TensorB= np.reshape(TensorA, [112, 512, 7, 7]), np.reshape(TensorB, [112, 512, 7, 7])
    # A1 = np.mean(np.mean(np.mean(TensorA, axis=2), axis=2), axis=1)
    # B1 = np.mean(np.mean(np.mean(TensorB, axis=2), axis=2), axis=1)
    # device = 'cpu'
    # TensorA, TensorB = TensorA.to('cpu'), TensorB.to('cpu')
    TensorA, TensorB = np.array(TensorA), np.array(TensorB)
    A1 = np.mean(np.mean(TensorA, axis=2), axis=2)
    B1 = np.mean(np.mean(TensorB, axis=2), axis=2)
    MI_loss = 0
    # start = time.time()
    for i in range(A1.shape[1]):
        TempA = A1[:, i]
        TempB = B1[:, i]
        MI_loss += metrics.normalized_mutual_info_score(TempA, TempB)
    out = MI_loss / A1.shape[1]
    # print(time.time() - start)

    # start = time.time()
    # out = NMI(A1, B1)
    # print(time.time()-start)
    # start = time.time()
    # metrics.normalized_mutual_info_score(A1, B1)
    # print(time.time()-start)
    return torch.from_numpy(np.asarray(out))


def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

if __name__ == '__main__':
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
    print(NMI(A,B))
    print(metrics.normalized_mutual_info_score(A,B))

    result = MILoss()
    print(result)