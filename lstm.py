import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, input):
        output, (h_n, c_n) = self.LSTM(input)

        last_timestep = output[:, -1, :]
        x = self.linear(last_timestep)
        return x