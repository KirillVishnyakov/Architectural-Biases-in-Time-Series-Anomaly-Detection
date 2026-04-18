import torch.nn as nn
import utils.config as config
from utils.RevIN import RevIN

class lstm_forecaster(nn.Module):
    """ Implements the lstm forecaster

    Args
    ---------
    hidden_size : int 
        lstm's hidden size
    horizon : int
        number of timesteps in the future the model predicts (contiguous)
    num_layers : int
        number of lstm layers
    input_size : int
        number of expected features in the input
    output_size : int
        number of expected features in the output

    Example
    ---------
    >>> model = lstm_forecaster(hidden_size = 128, horizon = 4, num_layers = 1)
    """

    def __init__(
        self, 
        hidden_size, 
        horizon = 1, 
        num_layers = 1, 
        input_size = 17, 
        output_size = 17
    ):
        super().__init__()        
        self.horizon = horizon
        self.output_size = output_size
        self.revin_layer = RevIN(input_size)
        self.LSTM = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True, 
            num_layers = num_layers
        )
        
        self.linear = nn.Linear(hidden_size, self.output_size * self.horizon)

    def forward(self, x):
        """ Returns the lstm's predicted horizon

        Args
        ---------
        x : tensor (batch, seq_len, feature) -> (B, L, M)

        Returns
        ---------
        tensor (batch, horizon, feature)
        
        Example
        ---------
        >>> x = torch.randn(32, 100, 17)
        >>> prediction = model(x)
        >>> prediction.shape
        torch.Size([32, 4, 17])
        """
        x = self.revin_layer(x, 'norm')
        output, _ = self.LSTM(x)

        last_timestep = output[:, -1, :]
        x = self.linear(last_timestep)
        x = x.view(-1, self.horizon, self.output_size)
        return self.revin_layer(x, 'denorm')

    