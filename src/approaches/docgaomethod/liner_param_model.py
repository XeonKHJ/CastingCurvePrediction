import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class LinerParamModel(nn.Module):
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
        self.relu1 = nn.ReLU()
        self.bp2 = nn.Linear(4, 4)
        self.relu2 = nn.ReLU()
        self.bp3 = nn.Linear(4, 2)


    def forward(self, x):
        x = self.bp1(x)
        x = self.relu1(x)
        x = self.bp2(x)
        x = self.relu2(x)
        x = self.bp3(x)
        return x

                