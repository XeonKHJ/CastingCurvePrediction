from numpy import quantile
import torch
import torch.nn as nn
import os
import math
from castingPredictModel import CastingPredictModel

def enrichData(dataset, window1, window2):
    step1step = window1/2
    step2step = window2/2

    enrichS = list()
    for data in dataset:
        enrichPerData = list()
        index = 0
        while index < data.__len__() - window1:
            dataInWindow = data[index:index+window1]
            windowDor = torch.sqrt(torch.sum(torch.pow(dataInWindow, 2)))
            enrichPerData.append(windowDor)
            index += int(step1step)
        enrichS.append(enrichPerData)

    enrichSTensor = torch.tensor(enrichS)
    enrichS = list()
    for data in enrichSTensor:
        enrichPerData = list()
        index = 0
        while index < data.__len__() - window2:
            dataInWindow = data[index:index+window2]
            mean = torch.mean(dataInWindow)
            max = torch.max(dataInWindow)
            min = torch.min(dataInWindow)
            quantile75 = torch.quantile(dataInWindow, 0.75)
            quantile5 = torch.quantile(dataInWindow, 0.5)
            quantile25 = torch.quantile(dataInWindow, 0.25)
            std = torch.std(dataInWindow)
            ptp = max - min
            newData = list([mean, max, min, quantile75, quantile5, quantile25, std, ptp])
            enrichPerData.append(newData)
            index += int(step2step)
        enrichS.append(enrichPerData)

    resultTensor = torch.tensor(enrichS)
    print("Dataset enriched")
    return resultTensor



def ReadData():
    datasetFolder = "../../../datasets/Datas/"
    allfiles = os.listdir(datasetFolder)
    stage1 = list()
    stage2 = list()
    allStage = list()
    # allStage.append(stage1)
    # allStage.append(stage2)
    
    allfiles = list(["test.csv"])

    for file in allfiles:
        stage1f = list()
        stage2f = list()
        datafile = open(file, encoding='utf-8')
        lines = datafile.readlines()

        datasetNum = 7
        indexToStart = 0
        # while float(lines[indexToStart].split(';')[5]) == 0:
        #     indexToStart += 1

        for i in range(datasetNum):
            stage1f.append(list())
            stage2f.append(list())

        attrsList = list() 
        for i in range(151):
            if i == 0:
                continue
            line = lines[i + indexToStart]
            attris = line.split(',')
            attrPerFile = list()
            for j in range(0,datasetNum,1):
                stage1f[j].append(float(attris[j]))

        indexToStart += 150
        for i in range(150):
            line = lines[i+indexToStart]
            attris = line.split(',')
            attrPerFile = list()
            for j in range(0,datasetNum,1):
                stage2f[j].append(float(attris[j]))
        allStage.append(stage1f)
        #enrichData(stage1f, 4, 2)
        allStage.append(stage2f)

    return allStage


def readConfig(enrich = True):
    hiddenLayerSize = 100
    if enrich:
        attrs = list(
            [list([8,hiddenLayerSize,3]),
            list([8,3,3]),]
        )
    else:
        attrs = list(
            [list([8,hiddenLayerSize,3]),
            list([1,3,3]),]
        )
    enrich = True


# hz
sampleRate = 2

if __name__ == '__main__':
    
    config = readConfig()

    print ('now __name__ is %s' %__name__)
    allStage = ReadData()
    timeLimitForEveryStep = list([1,2,3,4,5,6])
    border = 6
    # extract trainning and vaidation sets.
    trainningSet = list([allStage[0][0:border], allStage[1][0:border]])
    validationSet = list([allStage[0][border:7], allStage[1][border:7]])

    #inputsize(feature size) hidden_size(LSTM output size)

    # attr for every stage LSTM.
    # order: 0. feature size, 1. hidden size (LSTM output feature size), 2. number of hidden layers
    
    hiddenLayerSize = 100
    attrs = list(
        [list([8,hiddenLayerSize,3]),
        list([8,3,3]),]
    )

    feature_nums = list([1,1])
    
    lstm_model = CastingPredictModel(attrs,hiddenLayerSize,1)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

    # reshape tensor to (batch size, time series length, feature size)
    trainningTensor = torch.tensor(trainningSet[0]).reshape([trainningSet[0].__len__(),-1,1])
    enrichedTrainningTensor = enrichData(trainningTensor, 4, 2)
    yTensor = torch.tensor(trainningSet[1]).reshape([trainningSet[1].__len__(),-1,1])
    validationTensor = torch.tensor(validationSet[0]).reshape([validationSet[0].__len__(),-1,1])
    enrichedValidationTensor = enrichData(validationTensor, 4, 2)
    validationYTensor = torch.tensor(validationSet[1]).reshape([validationSet[1].__len__(),-1,1])
    max_epochs = 10000000
    for epoch in range(max_epochs):
        output = lstm_model(enrichedTrainningTensor)
        loss = loss_function(output, yTensor.reshape([yTensor.shape[0],-1]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print(loss)
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
            val = lstm_model(enrichedValidationTensor)
            print('validation loss:{}'.format(loss_function(val, validationYTensor.reshape([validationYTensor.shape[0], -1]))))

