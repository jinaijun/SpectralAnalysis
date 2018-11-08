#!/usr/bin/env python
#_*_ coding:utf-8 _*_

import os, csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 扫描已经测得的光谱库文件
# 将其取出至data数据矩阵
rootDir = 'D:\work\program\spectra analysis\data'   
fileList = os.listdir(rootDir)
data = np.zeros([len(fileList), 1805])   
for i in range(len(fileList)):
    filePath = os.path.join(rootDir, fileList[i])
    print(filePath)
    if os.path.isfile(filePath):
        csvFile = open(filePath)
        reader = csv.reader(csvFile)
        data1 = np.array(list(reader)[215:2020])    
        data[i,:] = list(map(float, data1[:,6]))

# 数据进行标准化处理
ss = StandardScaler().fit(data)
dataSS = ss.transform(data)
# PCA
pca = PCA(n_components=4)   # PCA参数
pca.fit(dataSS)
dataTran = pca.fit_transform(dataSS)    # 原始光谱数据矩阵经PCA变换后的数据
print(pca.components_)  # 打印PCA参数 
print(np.dot(dataSS, pca.components_.T))    # 使用PCA变换矩阵即截取的特征向量矩阵对原始光谱数据矩阵进行变换 
print(pca.singular_values_) # PCA特征值
print(pca.explained_variance_ratio_)    # 特征值百分比
print(dataTran)
# 画图
color4marker = ['b', 'r', 'k', 'g']
markers = ['o', '^', 's', 'p']
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection = '3d')
for k in range(28):
    ax.scatter(dataTran[k,0], dataTran[k,1], dataTran[k,2], c = color4marker[k//7], marker=markers[k//7])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
# HCA
dataHca = linkage(dataTran, 'ward')
fig2 = plt.figure()
dendrogram(dataHca)
plt.show()

# 获取最新样本光谱
newRootDir = 'D:\work\program\spectra analysis\data'   
newFileList = os.listdir(newRootDir)
newData = np.zeros([1, 1805])    
newFilePath = os.path.join(newRootDir, newFileList[0])
print(newFilePath)
if os.path.isfile(newFilePath):
    newCsvFile = open(newFilePath)
    newReader = csv.reader(newCsvFile)
    newData1 = np.array(list(newReader)[215:2020])    
    newData[0,:] = list(map(float, newData1[:,6]))
    print(newData)
# 对新样本进行标准化
newDataSS = ss.transform(newData)
# 对新样本使用获得的PCA model进行处理
newDataTran = np.dot(newDataSS, pca.components_.T)
print(newDataTran)
# 将新样本与光谱库合并
newData4HCA = np.vstack((dataTran, newDataTran))
# 对整个数据进行HCA计算
newDataHca = linkage(newData4HCA, 'ward')
# 画HCA图
fig3 = plt.figure()
dendrogram(newDataHca)
plt.show()