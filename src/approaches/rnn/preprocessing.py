import os
import sys
from typing import Counter, ForwardRef
import numpy
from numpy.lib.function_base import append
from numpy.lib.npyio import fromregex
import pandas

allfiles = 0
def PreReadData():
    datasetFolder = "../../../Datasets/fuck/"
    allfiles = os.listdir(datasetFolder)

    allStage = list()
    
    for file in allfiles:
        stage1f = list()
        stage2f = list()
        datafile = open(datasetFolder + file, encoding='utf-8')
        lines = datafile.readlines()

        indexToStart = 1
        while float(lines[indexToStart].split(';')[5]) == 0:
            indexToStart += 1

        attrsList = list() 
        for i in range(lines.__len__() - indexToStart):
            if i == 0:
                continue
            line = lines[i + indexToStart]
            attris = line.split(';')
            attrsList.append(float(attris[5]))
        allStage.append(attrsList)
    return allStage

# return a array that contains following info:
# smallestComp[dataIndex] = index of ogData that has smallest MSE with smallestComp[dataIndex]
def MoveFowardBy(data,ogData,step):
    ogLen = ogData.__len__()
    len = data.__len__()
    smallestComp = list()
    for i in range (int(ogLen/step)):
        smallestComp.append(0.0)

    if step > 0:
        for i in range(0, len, step):
            sublist2 = data[i:step+i]
            minMse = sys.float_info.max
            minIndex = int(len/step)
            for j in range(0, ogLen, step):
                ogSublist = ogData[j:(step + j)]
                mse = mseData(sublist2, ogSublist)
                if mse < minMse:
                    minMse = mse
                    minIndex = int(j/step)           
            smallestComp[int(i/step)] = list()
            smallestComp[int(i/step)].append(minIndex)  
            smallestComp[int(i/step)].append(minMse)
    return smallestComp
            
            
           
def mseData(data1, data2):
    data1Np = numpy.array(data1)
    data2Np = numpy.array(data2)
    mse = (numpy.square(data1Np - data2Np)).mean()
    return mse

def writeFile(dataset):
    datasetDict = dict()
    for i in range(len(dataset)):
        datasetDict[('value'+str(i))] = dataset[i]
    dataframe = pandas.DataFrame(datasetDict)
    dataframe.to_csv("test.csv",index=False,sep=',')


datasetLength = 2500

step = 100
dataset = PreReadData()
startindice = list()
for data in dataset:
    for dataIndice in range(len(data)):
        if data[dataIndice] > 5:
            startindice.append(dataIndice)
            break

for i in range(len(dataset)):
    dataset[i] = dataset[i][startindice[i] - 50: startindice[i] + datasetLength - 50]



# minMse = 0
writeFile(dataset)

print(dataset)
