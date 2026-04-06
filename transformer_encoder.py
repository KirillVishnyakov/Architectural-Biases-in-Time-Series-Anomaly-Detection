import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import config as config
from RevIN import RevIN

class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)
        
def init_weights_xavier(module):
    """Apply Xavier initialization to linear layers and embeddings with small gain for stability."""
    if isinstance(module, nn.Linear):
        # Use smaller gain for better stability
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal init for embeddings is more stable
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

class patch_transformer_encoder(nn.Module):
    def __init__(self, lookback_window = 100, forecast_horizon = 4, d_model = 256, nhead = 8, dropout = 0.1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.dropout = dropout
        
        self.patch_length = 16
        self.stride = 8
        n_patches = math.floor((lookback_window - self.patch_length) / self.stride) + 2
        print(n_patches)

        self.feature_mixer = nn.Sequential(
            nn.Linear(17, 64),
            nn.ReLU(),
            nn.Sequential(64, 17)
        )
        self.revin_layer = RevIN(17)
        self.input_proj = nn.Linear(self.patch_length, d_model, dtype=torch.float32)
        self.pos_embed = positional_encoding(d_model, dropout = self.dropout)

        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = d_model, nhead = nhead, batch_first = True, dropout = self.dropout, dtype=torch.float32)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4, norm = encoder_norm)
        
        self.linear_end = nn.Linear(n_patches * d_model, forecast_horizon, dtype=torch.float32)

        
        self.apply(init_weights_xavier)
        
    
    def forward(self, x):
        #x is [B, L, M]
        B, L, M = x.shape
        x = self.feature_mixer(x)
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1) #following papers structure
        #[B, M, L]
        x = x.reshape(B * M, L) #if deleguate features to batch, model processes 1 feature at a time for each batch -> vectorized channel independent inputs
        #[B*M, L]
        x = self.patch(x)
        #[B*M, N, P]
        x = self.input_proj(x)
        #[B*M, N, P] @ [P, D] = [B*M, N, D] which is number of patches x dimension
        x = self.pos_embed(x)
        x = self.encoder(x)
        #[B*M, N, D]
        x = x.flatten(1)
        #[B*M, N x D]
        x = self.linear_end(x)
        #[B*M, forecast_horizon]
        x = x.reshape(B, M, -1) #[B, M, forecast_horizon]
        x = torch.transpose(x, 1, 2) # [B, horizon, M]
        return self.revin_layer(x, 'denorm')
    def patch(self, x):
        #input is [B*M, L]
        #pad with last value just enough to guarantee unfolding every value
        x = F.pad(x, pad = (0, self.stride), mode = "replicate")
        x = x.unfold(dimension = 1, size = self.patch_length, step = self.stride)
        #x is [B*M, N, P]
    
        return x