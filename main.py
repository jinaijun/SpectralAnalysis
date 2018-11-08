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


ss = StandardScaler().fit(data)
dataSS = ss.transform(data)

pca = PCA(n_components=4)   
pca.fit(dataSS) 
dataTran = pca.fit_transform(dataSS)    
print(pca.components_)  
print(np.dot(dataSS, pca.components_.T))    
print(pca.singular_values_)
print(pca.explained_variance_ratio_)    
print(dataTran) 

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

dataHca = linkage(dataTran, 'ward')

fig2 = plt.figure()
dendrogram(dataHca)
plt.show()


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
newDataSS = ss.transform(newData)
newDataTran = np.dot(newDataSS, pca.components_.T)
print(newDataTran)

newData4HCA = np.vstack((dataTran, newDataTran))
newDataHca = linkage(newData4HCA, 'ward')

fig3 = plt.figure()
dendrogram(newDataHca)
plt.show()