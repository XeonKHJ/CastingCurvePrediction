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

def writeFile(data1, dataOffset, length):
    dataframe = pandas.DataFrame({'value':data1[dataOffset:length]})
    dataframe.to_csv("test.csv",index=False,sep=',')


step = 100
data = PreReadData()
offset = 0
datasetLength = 1500
portion = 10

ogData = data[0][0:datasetLength]
cpData = data[4][offset:datasetLength + offset]
cpDataLen = cpData.__len__()
step = int(datasetLength / portion)
offseted = dict()
minPortionMse = sys.float_info.max
minOffset = 0


while True:
    maxCount = dict()
    maxCountWeight = dict()
    step = int(datasetLength / portion)
    datashit = MoveFowardBy(cpData, ogData, int(datasetLength / portion))
    
    maxStepCount = 0
    maxStep = 0
    stepSum = 0
    # calculate offset weight
    for i in range(datashit.__len__()):
        maxCount[(i*step)-(datashit[i][0])*step] = 0
        maxCountWeight[(i*step)-(datashit[i][0])*step] = 0 
    for i in range(datashit.__len__()):
        currentStep = (i*step)-(datashit[i][0])*step
        maxCount[currentStep] += 1
        stepSum += currentStep
        currentCount = maxCount[currentStep]
        if currentCount > maxStepCount:
            maxStepCount = currentCount
            maxStep = currentStep
        maxCountWeight[currentStep] += datashit[i][1]
    
    minWeightStep = 0
    maxWeightStep = 0
    minWeight = sys.float_info.max
    maxWeight = 0
    for i in maxCount.keys():
        maxCountWeight[i] /= maxCount[i]
        if maxCountWeight[i] < minWeight:
            minWeight = maxCountWeight[i]
            minWeightStep = i
        if maxCountWeight[i] > maxWeight:
            maxWeight = maxCountWeight[i]
            maxWeightStep = i

    offset +=  maxStep
    realOffset = offset
    ogData = data[0][0:datasetLength]
    cpData = data[4][realOffset:datasetLength+realOffset]
    if realOffset in offseted:
        # portion *= 2
        step = int(datasetLength / portion)
        #offset = 0
        portionMse = mseData(cpData, ogData)
        if portionMse < minPortionMse:
            minPortionMse = portionMse
        # else:
        #     break
        offseted = dict()
    else:
        offseted[realOffset] = True

if offset > 0:
    while (datasetLength+realOffset) < (cpDataLen - 2):
        cpSubste = cpData[realOffset:datasetLength+realOffset]
        mse = mseData(ogData, cpSubste)
        if mse < minPortionMse:
            minPortionMse = mse
            minOffsetm = realOffset
        realOffset += 1
elif offset < 0:
    while (datasetLength+realOffset) < (cpDataLen - 2):
        cpSubste = cpData[realOffset:datasetLength+realOffset]
        mse = mseData(ogData, cpSubste)
        if mse < minPortionMse:
            minPortionMse = mse
            minOffsetm = realOffset
        realOffset += 1

# minMse = 0
writeFile(cpData, minOffsetm, datasetLength)

print(data)
