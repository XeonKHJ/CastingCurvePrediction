import torch
import torch.nn as nn
import os
from castingPredictModel import CastingPredictModel
from autoEncoderModel import CastingEncoderModel
from datetime import datetime, timedelta
import pandas
import requests, json


def preProcessData(dataset, window1, window2):
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
    #print("Dataset enriched")
    return resultTensor


def ReadContext():
    datasetFolder = "C:/Users/redal/source/repos/CastingCurvePrediction/src/approaches/rnn/"
    file = datasetFolder + "context.csv"
    datafile = open(file, encoding='utf-8')
    lines = datafile.readlines()
    context = dict()
    for i in range(1, 5):
        attris = lines[i].split(',')
        context[attris[0]] = list({int(attris[3]) / 1000000,int(attris[6]) / 100})
    abcdsdf = 'sdfsdf'
    return context
        



def ReadData(stages):
    datasetFolder = "C:/Users/redal/source/repos/CastingCurvePrediction/src/approaches/rnn/"
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
        datafile = open(datasetFolder + file, encoding='utf-8')
        lines = datafile.readlines()

        datasetNum = 7
        indexToStart = 0
        # while float(lines[indexToStart].split(';')[5]) == 0:
        #     indexToStart += 1

        for i in range(datasetNum):
            stage1f.append(list())
            stage2f.append(list())

        attrsList = list() 
        for i in range(stages[0]):
            if i == 0:
                continue
            line = lines[i + indexToStart]
            attris = line.split(',')
            attrPerFile = list()
            for j in range(0,datasetNum,1):
                stage1f[j].append(float(attris[j]))

        indexToStart += stages[0]
        for i in range(stages[1]):
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


def shuffle():
    border = 3
    allNum = 7
    # extract trainning and vaidation sets.
    stage1tranSetFront = allStage[0][0:border] + allStage[0][border+1:allNum]
    trainningSet = list([allStage[0][0:border], allStage[1][0:border]])
    validationSet = list([allStage[0][border:7], allStage[1][border:7]])
    trainningTensor = torch.tensor(trainningSet[0]).reshape([trainningSet[0].__len__(),-1,1])
    yTensor = torch.tensor(trainningSet[1]).reshape([trainningSet[1].__len__(),-1,1])
    trainningTensor = preProcessData(trainningTensor, 4, 2)
    validationTensor = torch.tensor(validationSet[0]).reshape([validationSet[0].__len__(),-1,1])
    validationYTensor = torch.tensor(validationSet[1]).reshape([validationSet[1].__len__(),-1,1])
    validationTensor = preProcessData(validationTensor, 4, 2)

def uploadStatus(date, losses, taskId):
    url_json = 'http://localhost:8080/UploadLosses'
    data_json = json.dumps({'taskId':taskId,'losses':losses})   #dumps：将python对象解码为json数据
    r_json = requests.post(url_json,json={'taskId':taskId,'losses':losses, 'times': date})
    print(r_json)

# hz
sampleRate = 2
taskId = 29
if __name__ == '__main__':
    print ("dada")
    config = readConfig()

    print('now __name__ is %s' %__name__)
    context = ReadContext()
    stageSpliter = [41,150]
    allStage = ReadData(stageSpliter)
    timeLimitForEveryStep = list([1,2,3,4,5,6])
    
    #inputsize(feature size) hidden_size(LSTM output size)
    mTimes = list()
    mLosses = list()

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

    # ++++++++++++ try auto-encoder +++++++++++++++++++++++
    autoencoder = CastingEncoderModel()
    autoencoderLoss = nn.MSELoss()
    autoencoderOptimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-2)

    length = 6
    border = 3
    trainningSet = list([allStage[0][0:border] + allStage[0][border+1:length], allStage[1][0:border] + allStage[1][border+1:length]])
    validationSet = list([allStage[0][border:border+1], allStage[1][border:border+1]])
    trainningFirstTensor = torch.tensor(trainningSet[0])
    trainningSecondTensor = torch.tensor(trainningSet[1])
    toBeEncodedTensor = torch.cat([trainningFirstTensor, trainningSecondTensor], 1).reshape([trainningFirstTensor.shape[0], -1, 1])
    yTensor = torch.tensor(trainningSet[1]).reshape([trainningSet[1].__len__(),-1,1])
    # trainningTensor = preProcessData(trainningTensor, 4, 2)
    validationFirstTensor = torch.tensor(validationSet[0])
    validationSecondTensor = torch.tensor(validationSet[1])
    tobeValidateTensor = torch.cat([validationFirstTensor, validationSecondTensor], 1).reshape([validationFirstTensor.shape[0], -1, 1])
    allTensor = torch.cat([torch.tensor(allStage[0]), torch.tensor(allStage[1])], 1)
    allTensor = allTensor.reshape([allTensor.shape[0], -1, 1])
    epoch = 0
    while True:
        epoch += 1
        output = autoencoder(toBeEncodedTensor)
        loss = autoencoderLoss(output, toBeEncodedTensor)
        loss.backward()
        autoencoderOptimizer.step()
        autoencoderOptimizer.zero_grad()
        mTimes.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        mLosses.append(loss.item())
        if epoch % 10 == 0 :
            print("{},{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), loss.item()))
        if epoch % 10 == 0:
            uploadStatus(mTimes, mLosses, taskId)
            mTimes.clear()
            mLosses.clear()
        if epoch > 200:
            output = autoencoder(allTensor)
            datasetDict = dict()
            for i in range(len(output)):
                datasetDict[('value'+str(i))] = output[i].reshape([-1]).tolist()
            dataframe = pandas.DataFrame(datasetDict)
            dataframe.to_csv("autoencoderresult.csv",index=False,sep=',')
            break

            

    # # validationTensor = preProcessData(validationTensor, 4, 2)

        
    

    # # ------------ try auto-encoder -----------------------

    # # ++++++++++++ generate first phase +++++++++++++++++++
    
    # contextList = list()
    # for i in context.values():
    #     contextList.append(i)
    
    # contextTensor = torch.tensor(contextList).reshape([contextList.__len__(), -1, 2]).float()
    # # yTensor = torch.tensor(allStage[0][0:4]).reshape([contextList.__len__(), 1, -1])
    # # while(True):   
    # #     output = lstm_model.firstStageForward(contextTensor)
    # #     loss = loss_function(output, yTensor)
    # #     print(loss)
    # #     loss.backward()
    # #     optimizer.step()
    # #     optimizer.zero_grad()

    # # ------------------------end---------------------------
        
    





    # max_epochs = 10000000
    # border = 3
    # timeAndLoss = list([list(),list()])
    # now = datetime.now()
    # for epoch in range(max_epochs):
    #     if (epoch % 100 == 0):
    #         length = 4
    #         # extract trainning and vaidation sets.
    #         trainningSet = list([allStage[0][0:border] + allStage[0][border+1:length], allStage[1][0:border] + allStage[1][border+1:length]])
    #         validationSet = list([allStage[0][border:border+1], allStage[1][border:border+1]])
    #         trainningTensor = torch.tensor(trainningSet[0]).reshape([trainningSet[0].__len__(),-1,1])
    #         yTensor = torch.tensor(trainningSet[1]).reshape([trainningSet[1].__len__(),-1,1])
    #         trainningTensor = preProcessData(trainningTensor, 4, 2)
    #         validationTensor = torch.tensor(validationSet[0]).reshape([validationSet[0].__len__(),-1,1])
    #         validationYTensor = torch.tensor(validationSet[1]).reshape([validationSet[1].__len__(),-1,1])
    #         validationTensor = preProcessData(validationTensor, 4, 2)
    #         trainningContextSet = contextList[0:border]
    #         trainningContextTensor = torch.Tensor(trainningContextSet)
    #         validationContextSet = contextList[border:length]
    #         validationContextTensor = torch.tensor(validationContextSet)
    #         #border = 6
    #         print("shuffle!")
    #     output = lstm_model(trainningTensor, torch.tensor(trainningContextSet))
    #     loss = loss_function(output, yTensor.reshape([yTensor.shape[0],-1]))
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     timeAndLoss[0].append((datetime.now() - now).total_seconds())
    #     timeAndLoss[1].append(loss.item())
    #     #print(loss)
    #     if loss.item() < 1e-3:
    #         # print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
    #         # print("The loss value is reached")
    #         #break
    #         abc = 1 # do nothing
    #         break
    #     elif (epoch+1) % 10 == 0:     
    #         print('Epoch: [{}/{}], Loss:{}'.format(epoch+1, max_epochs, loss.item()))
    #         val = lstm_model(validationTensor, torch.tensor(validationContextSet))
    #         print('validation loss:{}'.format(loss_function(val, validationYTensor.reshape([validationYTensor.shape[0], -1]))))

    # timeAndLossDict = dict()
    # timeAndLossDict['timespan'] = timeAndLoss[0]
    # timeAndLossDict['loss'] = timeAndLoss[1]
    # dataframe = pandas.DataFrame(timeAndLossDict)
    # dataframe.to_csv("timeAndLoss.csv",index=False,sep=',')


    # lstm_model.eval()
    # # output eval data
    # predSet = list([allStage[0][0:length], allStage[1][0:length]])
    # predTensor = preProcessData(torch.tensor(predSet[0]).reshape([predSet[0].__len__(),-1,1]), 4, 2)
    # result = lstm_model(predTensor, torch.tensor(contextList))
    # result = result.tolist()
    
    # datasetDict = dict()
    # for i in range(len(result)):
    #     datasetDict[('value'+str(i))] = allStage[0][i] + result[i]
    # dataframe = pandas.DataFrame(datasetDict)
    # dataframe.to_csv("predData.csv",index=False,sep=',')
    

