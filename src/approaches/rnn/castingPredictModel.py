from multiprocessing import context
from turtle import back
from numpy import dtype, reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class CastingPredictModel(nn.Module):
    """
        Parameters：
        - feature_nums: list of feature num for every detector.
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, feature_nums, hidden_size=3600,  num_layers=1, context_feature=2):
        super().__init__()
        
        self.detectors = list()

        # input_size is output_size
        for i in range(feature_nums.__len__()):
            self.detectors.append(nn.LSTM(feature_nums[i][0], feature_nums[i][1], feature_nums[i][2], batch_first = True))

        self.lstm = nn.LSTM(feature_nums[0][0], feature_nums[0][1], feature_nums[0][2], batch_first = True)
        self.forwardCalculation = nn.Linear(feature_nums[0][1],1)
        self.liner = nn.Linear(16+context_feature, 150)
        self.finalCalculation = nn.Sigmoid()

        #self.preLstm = nn.LSTM(context_feature.__len__(), 100, 40, batch_first = True)

        # 异常检测网络
        self.autoEncoder = {
            nn.Linear(1,1,1),
            nn.Linear(1,1,1),
            nn.Linear(2,2,2)
        }

        # Context
        self.preLstm = nn.LSTM(context_feature, 10, 10, batch_first = True)
        self.preLinear = nn.Linear(10, 40)

    def firstStageForward(self, _x):
        x = _x
        x, b = self.preLstm(x)
        x = self.preLinear(x)
        return x

    def forward(self, _x, context):
        # x = torchrnn.pack_padded_sequence(_x, xTimestampSizes, True)
        x = _x
        x, b = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        
        # x, xBatchSize = torchrnn.pad_packed_sequence(x, batch_first=True)

        # forward_to_stack_x = torch.transpose(x, 0, 1)

        x = self.forwardCalculation(x)
        x = x.reshape([x.shape[0],-1])
        x = torch.cat([x, context], 1)
        x = self.liner(x)
    

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

                