import torch.nn as nn
import utils.config as config
from utils.RevIN import RevIN

class lstm_forecaster(nn.Module):
    def __init__(self, hidden_size, horizon = 1, num_layers = 1, input_size = 17, output_size = 17):
        super().__init__()
        
        self.horizon = horizon
        self.output_size = output_size
        self.revin_layer = RevIN(input_size)
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers = num_layers)
        
        self.linear = nn.Linear(hidden_size, self.output_size * self.horizon)

    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        output, _ = self.LSTM(x)

        last_timestep = output[:, -1, :]
        x = self.linear(last_timestep)
        x = x.view(-1, self.horizon, self.output_size) #(batch, timesteps, features)
        return self.revin_layer(x, 'denorm')

    