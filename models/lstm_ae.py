import torch.nn as nn
from utils.RevIN import RevIN

class lstm_encoder(nn.Module):
    def __init__(self, num_features = 17, lookback_window = 50, latent_dim = 0.30):
        super(lstm_encoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(self.num_features, self.latent_dim, num_layers = 1, batch_first = True)
    
    def forward(self, x): # [B, L, M]
        if x.dim() < 3: x = x.unsqueeze(0)
        _, (h_n, _) = self.lstm(x) # [1, B, latent_dim]
        return h_n.squeeze(0) #[B, latent_dim]

class lstm_decoder(nn.Module):
    def __init__(self, latent_dim, lookback_window = 50, num_features = 17):
        super(lstm_decoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim

        self.initial_state = nn.Linear(self.latent_dim, self.num_features)
        self.initial_cell_state = nn.Linear(self.latent_dim, self.num_features)
        self.lstm = nn.LSTM(self.latent_dim, self.num_features, num_layers = 1, batch_first = True)
    def forward(self, x):  # [B, latent_dim]
        h0 = self.initial_state(x).unsqueeze(0) # [1, B, latent_dim] expected for pytorch lstm
        c0 = self.initial_cell_state(x).unsqueeze(0) # [1, B, latent_dim] expected for pytorch lstm

        # [B, latent_dim] -> [B, 1, latent_dim] -> [B, self.lookback_window, latent_dim]
        input = x.unsqueeze(1).repeat(1, self.lookback_window, 1)
        out, _ = self.lstm(input, (h0, c0))
        return out

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