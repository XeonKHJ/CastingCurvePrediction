from multiprocessing import context
from turtle import back
from numpy import dtype, reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class CastingEncoderModel(nn.Module):
    """
        Parameters：
        - feature_nums: list of feature num for every detector.
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self):
        super().__init__()
        
        self.detectors = list()

        self.encoder = nn.LSTM(1, 100, 5, batch_first = True)
        self.decoder = nn.LSTM(100, 10, 5, batch_first = True)

        self.forwardCalculation = nn.Linear(10,1)

    def encode(self, _x):
        x, b = self.encoder(_x)
        return x

    def decode(self, _x):
        x, b = self.decoder(_x)
        x = self.forwardCalculation(x)
        return x

    def forward(self, _x):
        x = _x
        x = self.encode(x)  
        x = self.decode(x)
        return x
    
    
    @staticmethod
    def PadData(dataLists, featureSize):
        # Sort data first
        dataLists.sort(key=(lambda elem:len(elem)), reverse=True)
        dataTimestampLengths = list()
        for i in range(len(dataLists)):
            dataTimestampLengths.append(len(dataLists[i]))
        

        # Padding data
        longestSeqLength = len(dataLists[0])
        dataBatchSize = len(dataLists)
        
        inputTensor = torch.zeros(dataBatchSize,longestSeqLength, featureSize).int()
        
        for i in range(dataBatchSize):
            currentTimeSeq = 1
            for j in range(len(dataLists[i])):
                inputTensor[i][j] = torch.tensor(dataLists[i][j])
       

        return inputTensor.float(), dataTimestampLengths

                