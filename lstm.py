import torch.nn as nn
import torch

class short_window_LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers = 1, input_size = 17, output_size = 17):
        super().__init__()


        self.LSTM = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers = num_layers)
        
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, (h_n, c_n) = self.LSTM(input)

        last_timestep = output[:, -1, :]
        x = self.linear(last_timestep)
        return x

# Can't use wrapper LSTM function, have to use LSTMCell to add dropout only to "g" gate, then perform the LSTM computations to get the final h_t.
class long_window_LSTM_recurrentDropout(nn.Module):
    def __init__(self, hidden_size, num_layers = 1, input_size = 17, output_size = 17, dropout = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.cells = nn.ModuleList([nn.LSTMCell(input_size if layer == 0 else hidden_size, hidden_size) 
                                    for layer in range(num_layers)])
        self.linear = nn.Linear(hidden_size, output_size)
        self.output_dropout = nn.Dropout(p = dropout)

    def forward(self, input):
        layer_input = input
        #input shape: [1, 100, 17]
        for idx, current_cell in enumerate(self.cells):
            #opposite of torch.cat([Wi, Wf, Wg, Wo], dim=0)
            Wii, Wif, Wig, Wio = current_cell.weight_ih.chunk(4, dim=0)
            Whi, Whf, Whg, Who = current_cell.weight_hh.chunk(4, dim=0)
            bii, bif, big, bio = current_cell.bias_ih.chunk(4, dim=0)
            bhi, bhf, bhg, bho = current_cell.bias_hh.chunk(4, dim=0)
            if idx == 0:
                h_t, c_t = torch.zeros(layer_input.shape[0], self.hidden_size, device=input.device), \
                           torch.zeros(layer_input.shape[0], self.hidden_size, device=input.device)

            #divide by / (1 - dropout) so training g's are the same scale as testing g's
            mask = torch.bernoulli(torch.full((layer_input.shape[0], self.hidden_size), 1 - self.dropout)\
                                   .float().to(layer_input.device)) / (1 - self.dropout)

            layer_output = []
            for time_step in range(input.shape[1]):
                x = layer_input[:, time_step, :]

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
            # layer_output is a list of 100 (batch_size, hidden_size) tensors
            # torch.stack transforms it to (batch_size x 100 x hidden_size) tensor
            layer_input = torch.stack(layer_output, 1)

        #python will set h_t to be the last timestep of the last layer by default
        out = self.linear(self.output_dropout(h_t))
        return out
    
