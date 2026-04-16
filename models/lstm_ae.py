import torch.nn as nn
from utils.RevIN import RevIN
import torch

class lstm_encoder(nn.Module):
    def __init__(self, num_features = 17, lookback_window = 50, latent_dim = 0.30):
        super(lstm_encoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(self.num_features, self.latent_dim, num_layers = 1, batch_first = True)
    
    def forward(self, x): # [B, L, M]
        if x.dim() < 3: x = x.unsqueeze(0)
        out, (final_hidden_State, final_cell_state) = self.lstm(x)
        return out, (final_hidden_State, final_cell_state), x #[B, L, latent_dim], [1, B, latent_dim], [1, B, latent_dim], true timestep

class lstm_decoder(nn.Module):
    def __init__(self, latent_dim, lookback_window = 50, num_features = 17):
        super(lstm_decoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim
        
        self.lstm = nn.LSTM(self.num_features + self.latent_dim, self.latent_dim, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(self.latent_dim, self.num_features)

    def forward(self, x):  # out, (final_hidden_State, final_cell_state)
        in_sequence, (h0, c0), true = x
        B, L, H = in_sequence.shape

        if self.training:
            true_sequences = true.roll(shifts = 1, dims = 1)
            true_sequences[:, 0:1, :] = torch.zeros((B, 1, self.num_features), device = in_sequence.device, dtype = in_sequence.dtype)
            decoder_input = torch.cat([true_sequences, in_sequence], dim = 2)

            reconstructed_window, _ = self.lstm(decoder_input, (h0, c0))
            reconstructed_window = self.linear(reconstructed_window)

        else:

            reconstructed_window = [] 
            start_token = torch.zeros((B, 1, self.num_features), device = in_sequence.device, dtype = in_sequence.dtype) 
            decoder_input = torch.cat([start_token, in_sequence[:, 0: 1, :]], dim = 2) 
            out, (hn, cn) = self.lstm(decoder_input, (h0, c0)) 
            out = self.linear(out) 
            reconstructed_window.append(out) 

            for i in range(1, L): 
                decoder_input = torch.cat([out, in_sequence[:, i: i + 1, :]], dim = 2) 
                out, (hn, cn) = self.lstm(decoder_input, (hn, cn)) # out is [B, 1, H] 
                out = self.linear(out) # out is [B, 1, M] 
                reconstructed_window.append(out) 

            reconstructed_window = torch.cat(reconstructed_window, dim=1) 

        return reconstructed_window

class lstm_ae(nn.Module):
    def __init__(self, num_features = 17, lookback_window = 50, embedding_dim_ratio = 0.30):
        super(lstm_ae, self).__init__()
        self.latent_dim = int(num_features * embedding_dim_ratio)
        self.encoder = lstm_encoder(num_features, lookback_window, self.latent_dim)
        self.revin_layer = RevIN(num_features)
        self.decoder = lstm_decoder(self.latent_dim, lookback_window, num_features)
    
    def forward(self, x):
        x = self.revin_layer(x, 'norm')
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.revin_layer(x, 'denorm')
        return x