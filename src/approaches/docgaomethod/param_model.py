import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class ParamModel(nn.Module):
    """
        Parameters:
        - feature_nums: list of feature num for every detector.
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self):
        super().__init__()
        hidden_size = 4
        num_layers = 10
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.bp1 = nn.Linear(3, 4)
        self.bp2 = nn.Linear(4, 4)
        self.bp3 = nn.Linear(4, 2)
        self.forwardCalculation = nn.Linear(hidden_size * num_layers,4)


    def forward(self, x):
        lstm_out, (h,c) = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        reshape_lstm_out = h.reshape([-1])
        foward_out = self.forwardCalculation(reshape_lstm_out)
        return foward_out

                