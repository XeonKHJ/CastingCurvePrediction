import torch
import torch.nn as nn
from castingPredictModel import CastingPredictModel

def ReadData():
    datasetFolder = "../../../Datasets/Datas"

    stage1 = list()
    stage2 = list()
    allStage = list()
    allStage.append(stage1)
    allStage.append(stage2)
    datafile = open(datasetFolder + '/data2.csv')
    lines = datafile.readlines()

    indexToStart = 1
    while float(lines[indexToStart].split(',')[1]) == 0:
        indexToStart += 1

    attrsList = list() 
    for i in range(150):
        if i == 0:
            continue
        line = lines[i + indexToStart]
        attris = line.split(',')
        stage1.append(float(attris[1]))

    indexToStart += 150
    for i in range(150):
        line = lines[i+indexToStart]
        attris = line.split(',')
        stage2.append(float(attris[1]))

    return allStage






# hz
sampleRate = 2

if __name__ == '__main__':
    
    print ('now __name__ is %s' %__name__)
    allStage = ReadData()
    timeLimitForEveryStep = list([1,2,3,4,5,6])


    #inputsize(feature size) hidden_size(LSTM output size)

    # attr for every stage LSTM.
    # order: 0. feature size, 1. hidden size (LSTM output feature size), 2. number of hidden layers
    attrs = list(
        [list([1,6,3]),
        list([1,3,3]),]
    )

    feature_nums = list([1,1])
    
    lstm_model = CastingPredictModel(attrs,200,1)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-1)
    
    # reshape tensor to (batch size, time series length, feature size)
    inputTensor = torch.tensor(allStage[0]).reshape([1,-1,1])
    yTensor = torch.tensor(allStage[1])
    max_epochs = 10000
    for epoch in range(max_epochs):
        output = lstm_model(inputTensor)
        loss = loss_function(output, yTensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))

