import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class OutlierDetectModel(nn.Module):
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
        num_layers = 5
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.forwardCalculation = nn.Linear(hidden_size * num_layers, 4)

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        lstm_out, (h,c) = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        reshape_lstm_out = h.reshape([-1])
        foward_out = self.forwardCalculation(reshape_lstm_out)
        return foward_out