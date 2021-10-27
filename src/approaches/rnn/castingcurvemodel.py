import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=3600,  num_layers=1):
        super().__init__()
        
        # input_size is output_size

        # forward LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first =True) # utilize the LSTM model in torch.nn 
        
        # reveresd LSTM
        self.rlstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        self.forwardCalculation = nn.Linear(12,6)
        self.finalCalculation = nn.Sigmoid()

    def forward(self, _x, xTimestampSizes):
        x = torchrnn.pack_padded_sequence(_x, xTimestampSizes, True)
        x, b = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        
        x, xBatchSize = torchrnn.pad_packed_sequence(x, batch_first=True)

        forward_to_stack_x = torch.transpose(x, 0, 1)

    

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
                