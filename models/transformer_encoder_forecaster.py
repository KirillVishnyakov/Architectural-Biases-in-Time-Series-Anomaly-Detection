import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.config as config
from utils.RevIN import RevIN

class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, D]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x is [B, M, N, D]
        T = x.size(2) # should be across N
        x = x + self.pe[:, :T, :].unsqueeze(1)
        return self.dropout(x)
         
class patch_module(nn.Module):
    def __init__(self, patch_length, stride):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
    
    def forward(self, x):
        # x is [B, L, M]
        x = x.permute(0, 2, 1) # x is [B, M, L]

        # now need to separate each L into a patches

        #pad L with last value stride times just enough to guarantee unfolding everything inside L (or else pytorch cuts incomplete strides)
        x = F.pad(x, pad = (0, self.stride), mode = "replicate")
        x = x.unfold(dimension = 2, size = self.patch_length, step = self.stride)
        """
        [B, M, L] -> [B, M, N, P]
        L -> [N, P] where N is num of patches, and P is patch length
        """
        return x
class structured_attention(nn.Module):
    def __init__(self, num_features = 17, penalty = -2.0):
        super().__init__()
        self.attn_bias = nn.Parameter(torch.zeros(num_features, num_features))

        command_idx = [0, 1, 5, 6]
        environmental_stimuli_idx = [2, 3, 4]
        system_triggers_dx = command_idx + environmental_stimuli_idx

        system_response_idx = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        with torch.no_grad():
            # commands and environment stimulies shouldnt attend to system responses
            for i in system_triggers_dx:
                for j in system_response_idx:
                    self.attn_bias[i, j] = penalty
            
            # commands and environmental stimulies also shouldnt interact much with eachother
            for i in command_idx:
                for j in environmental_stimuli_idx:
                    self.attn_bias[i, j] = penalty
                    self.attn_bias[j, i] = penalty
            
            # model can still learn to change how each feature attends to another.
    def forward(self):
        return self.attn_bias

class attention_module(nn.Module):
    def __init__(self, lookback_window = 100, d_model = 256, nhead = 8, dropout = 0.1, attention_bias = None):
        super().__init__()
        self.lookback_window, self.d_model, self.nhead, self.dropout = \
            lookback_window, d_model, nhead, dropout

        self.attention_bias = attention_bias
        
        self.feature_attention = self.encoder_template()
        self.patch_attention = self.encoder_template()

    def encoder_template(self):
        encoder_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.nhead, batch_first = True, dropout = self.dropout, dtype=torch.float32)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers = 1, norm = encoder_norm)
        return encoder

    def forward(self, x):
        B, M, N, D = x.shape #x is [B, M, N, D]
        """
        need to attend across channels, i.e
        [B, M, 1, D], [B, M, 2, D], [B, M, ..., D], [B, M, N, D] and attend them together

        for n in range(N):
            x_n = x[:, :, n, :]      # [B, M, D]  - M tokens, each of dim D
            Q = x_n @ W_Q             # [B, M, d_k]
            K = x_n @ W_K             # [B, M, d_k]
            V = x_n @ W_V             # [B, M, d_k]
            attn = softmax(Q @ K.T / sqrt(d_k)) @ V  # [B, M, d_k]

        but looping is too slow, so instead reshape [B, M, N, D] -> [B * N, M, D] and then attend
        """

        x = x.permute(0, 2, 1, 3) # [B, N, M, D]
        x = x.reshape(B * N, M, D) # [B * N, M, D]
        x = self.feature_attention(x, mask = self.attention_bias()) # [B * N, M, D]
        x = x.reshape(B, N, M, D)
        x = x.permute(0, 2, 1, 3) # [B, M, N, D]

        """
        each M token now has been cross attended with each other M token and possibly itself.
        """
        x = x.reshape(B * M, N, D) # [B * M, N, D]
        x = self.patch_attention(x) # [B * M, N, D]
        x = x.reshape(B, M, N, D) # [B, M, N, D]
        """
        each N token now has been cross attended with each other N.
        """
        return x

class patch_transformer(nn.Module):
    def __init__(self, lookback_window = 100, forecast_horizon = 4, d_model = 256, nhead = 8, dropout = 0.1, num_features = 17, num_blocks = 3):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.dropout = dropout
        self.patch_length = 16
        self.stride = 8
        self.patch = patch_module(self.patch_length, self.stride)
        n_patches = math.floor((lookback_window - self.patch_length) / self.stride) + 2
        print(n_patches)

        self.revin_layer = RevIN(num_features)
        self.input_proj = nn.Linear(self.patch_length, d_model, dtype=torch.float32)
        self.pos_embed = positional_encoding(d_model, dropout = dropout)
        self.feature_embed = nn.Parameter(torch.randn(1, num_features, 1, d_model) * 0.02)

        self.attention_bias = structured_attention(num_features = num_features, penalty = -1.0)

        self.blocks = nn.ModuleList([
            attention_module(lookback_window = 100, d_model = d_model, nhead = nhead, dropout = dropout, attention_bias = self.attention_bias) \
                for _ in range(num_blocks)])

        self.linear_end = nn.Linear(n_patches * d_model, forecast_horizon, dtype=torch.float32)
        
    def forward(self, x):
        #x is [B, L, M]
        B, L, M = x.shape
        x = self.revin_layer(x, 'norm')

        x = self.patch(x)
        # x is [B, M, N, P]

        #each patch is projected in the space needed for first attention layer
        x = self.input_proj(x) #[B, M, N, P] @ [P, D] = [B, M, N, D] 
        x = x + self.feature_embed
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x) # [B, M, N, D] 
        
        x = x.flatten(2) # [B, M, N * D]
        # [B, M, N * D] -> [B, M, horizon]

        x = self.linear_end(x) # [B, M, horizon]

        x = torch.transpose(x, 1, 2) # [B, horizon, M] to match input [B, L, M]
        return self.revin_layer(x, 'denorm')