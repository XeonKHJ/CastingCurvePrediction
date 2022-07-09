import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn
import torch.optim as optim

class EnrichTimeSeriesDataPreprocessor():
    def process(data):
        