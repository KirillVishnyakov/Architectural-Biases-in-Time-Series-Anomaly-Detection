import torch.nn as nn
import torch
from typing import List

class LSTMBaseline(nn.Module):
    def __init__(self, hidden_size, l = 1, num_layers = 1, input_size = 17, output_size = 17):
        super().__init__()
        
        self.l = l
        self.output_size = output_size
        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers = num_layers)
        
        self.linear = nn.Linear(hidden_size, output_size * l)

    def forward(self, input):
        output, (h_n, c_n) = self.LSTM(input)

        last_timestep = output[:, -1, :]
        x = self.linear(last_timestep)
        return x.view(-1, self.l, self.output_size) #(batch, timesteps, features)


    
@torch.jit.script
def lstm_layer_forward(
    layer_input: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias_ih: torch.Tensor,
    bias_hh: torch.Tensor,
    mask: torch.Tensor,
    is_training: bool
) -> torch.Tensor:
    batch_size = layer_input.shape[0]
    hidden_size = weight_hh.shape[1]

    h_t = torch.zeros(batch_size, hidden_size, device=layer_input.device)
    c_t = torch.zeros(batch_size, hidden_size, device=layer_input.device)
    layer_output: List[torch.Tensor] = []


    """
    The following code does the exact same thing as:
    i = torch.sigmoid(x @ Wii.T + bii + h_t @ Whi.T + bhi)
    f = torch.sigmoid(x @ Wif.T + bif + h_t @ Whf.T + bhf)
    g = torch.tanh(x @ Wig.T + big + h_t @ Whg.T + bhg)
    if self.training:
        masked_g = mask * g
    else:
        masked_g = g
    o = torch.sigmoid(x @ Wio.T + bio + h_t @ Who.T + bho)
    c_t = f * c_t + i * masked_g
    h_t = o * torch.tanh(c_t)
    layer_output.append(h_t)
    
    """
    # precompute input projection for all timesteps at once [batch, seq, 4*hidden]
    input_proj = layer_input @ weight_ih.T + bias_ih

    for t in range(layer_input.shape[1]):

        gates = input_proj[:, t, :] + h_t @ weight_hh.T + bias_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        if is_training:
            g = g * mask

        c_t = f * c_t + i * g
        h_t = o * torch.tanh(c_t)
        layer_output.append(h_t)


    # layer_output is a list of 100 (batch_size, hidden_size) tensors
    # torch.stack transforms it to (batch_size x 100 x hidden_size) tensor
    return torch.stack(layer_output, dim=1)


class long_window_LSTM_recurrentDropout_jit(nn.Module):
    def __init__(self, hidden_size, num_layers=1, input_size=17, output_size=17, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, output_size)
        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        layer_input = x
        batch_size = x.shape[0]

        for cell in self.cells:
            mask = torch.ones(batch_size, self.hidden_size, device=x.device)
            if self.training and self.dropout_p > 0:
                mask = \
                torch.bernoulli(torch.full((batch_size, self.hidden_size), 1 - self.dropout_p, device=x.device)) / (1 - self.dropout_p)

            layer_input = lstm_layer_forward(
                layer_input,
                cell.weight_ih,
                cell.weight_hh,
                cell.bias_ih,
                cell.bias_hh,
                mask,
                self.training
            )

        h_last = layer_input[:, -1, :]
        return self.linear(self.output_dropout(h_last))