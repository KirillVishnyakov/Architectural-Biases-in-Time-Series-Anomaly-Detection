import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import utils.config as config
from utils.RevIN import RevIN

class positional_encoding(nn.Module):
    """ Implements the sinusoidal positional encoding module (Vaswani et al., 2017)
    This module generates fixed (non-learnable) positional encodings and adds
    them to the input tensor along the sequence dimension

    Args
    ---------
    d_model : int
        dimensionality of the model embeddings
    max_len : int
        max sequence length supported
    dropout : float
        dropout probability
    
    Example
    ---------
    >>> model = patch_transformer(
    ...     lookback_window = 256, 
    ...     forecast_horizon = 4, 
    ...     d_model = 256, 
    ...     nhead = 8, 
    ...     dropout = 0.0, 
    ...     num_features = 17, 
    ...     num_blocks = 1
    ... )
    >>> pe = model.pos_embed
    """

    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, D]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """ Applies sinusoidal positional encoding to input and returns transformed tensor

        Args
        ---------
        x: tensor (batch, feature, patch, d_model) -> (B, M, N, D)

        Returns
        ---------
        tensor (B, M, N, D)
            input but with positional encoding applied along N
        
        Example
        ---------
        >>> x = torch.randn(32, 17, N, 256)
        >>> pe(x).shape
        torch.size([32, 17, N, 256])
        """
        #x is [B, M, N, D]
        T = x.size(2) # should be across N
        x = x + self.pe[:, :T, :].unsqueeze(1)
        return self.dropout(x)
         
class patch_module(nn.Module):
    """ Implements the patching mechanism 
    Each time series window is broken into sequences of patches (per feature)
    using a sliding window (torch.unfold)

    Args
    ----------
    patch_length : int
        length of a each patch
    stride : int
        step size between each patching window

    Example
    ----------
    >>> model = patch_transformer(
    ...     lookback_window = 256, 
    ...     forecast_horizon = 4, 
    ...     d_model = 256, 
    ...     nhead = 8, 
    ...     dropout = 0.0, 
    ...     num_features = 17, 
    ...     num_blocks = 1
    ... )
    >>> patching_module = model.patch
    """

    def __init__(self, patch_length, stride):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
    
    def forward(self, x):
        """ Converts input window into patches

        Args
        ----------
        x : tensor (batch, seq_len, feature) -> (B, L, M)

        Returns
        ----------
        x : tensor (batch, feature, patch, patch_length) -> (B, M, N, P)

        Example
        ---------
        >>> x = torch.randn(32, 256, 17)
        >>> patching_module(x).shape
        torch.Size([32, 17, N, P])
        """
        x = x.permute(0, 2, 1) # [B, M, L]

        # now need to separate each L into a patches

        #pad L with last value stride times just enough to guarantee unfolding everything inside L
        x = F.pad(x, pad = (0, self.stride), mode = "replicate")
        x = x.unfold(dimension = 2, size = self.patch_length, step = self.stride)
        """
        [B, M, L] -> [B, M, N, P]
        L -> [N, P] where N is num of patches, and P is patch length
        """
        return x

class attention_module(nn.Module):
    """ Implements the feature attention + patch attention block

    Args
    ---------
    lookback_window : int
        number of past timesteps provided as input
    d_model : int
        embedding dimension used inside transformer blocks
    nhead : int
        num of attention heads in each transformer block
    dropout : float
        dropout probability

    Example
    ---------
    >>> model = patch_transformer(
    ...     lookback_window = 256, 
    ...     forecast_horizon = 4, 
    ...     d_model = 256, 
    ...     nhead = 8, 
    ...     dropout = 0.0, 
    ...     num_features = 17, 
    ...     num_blocks = 1
    ... )
    >>> attention_module = model.blocks[0] # grabs a full block
    """

    def __init__(
            self, 
            lookback_window = 100, 
            d_model = 256, 
            nhead = 8, 
            dropout = 0.1
        ):
        super().__init__()
        self.lookback_window, self.d_model, self.nhead, self.dropout = \
            lookback_window, d_model, nhead, dropout
        
        self.feature_attention = self.encoder_template()
        self.patch_attention = self.encoder_template()

    def encoder_template(self):
        """ Helper function that creates an attention module """
        encoder_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model = self.d_model, 
                            nhead = self.nhead, 
                            batch_first = True, 
                            dropout = self.dropout, 
                            dtype=torch.float32
                        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers = 1, norm = encoder_norm)
        return encoder

    def forward(self, x):
        """ Apply two stage attention, first feature wise then patch wise
        Feature attention (across M) independently for each position N
        Patch attention (across N) independently for each position M

        Args
        ---------
        x : tensor (batch, feature, num_patches, d_model) -> (B, M, N, D)

        Returns
        ---------
        tensor (B, M, N, D)
            Output tensor after both attention stages.

        Example
        ---------
        >>> x = torch.randn(32, 17, 16, 256)
        >>> out = attention_module(x)
        >>> out.shape
        torch.Size([32, 17, 16, 256])
        """
        B, M, N, D = x.shape #x is [B, M, N, D]

        """
        need to attend across channels, i.e
        [B, M, 1, D], [B, M, 2, D], [B, M, ..., D], [B, M, N, D] and attend them together
        instead of looping, reshape [B, M, N, D] -> [B * N, M, D], then use pytorch wrappers
        """
        x = x.permute(0, 2, 1, 3) # [B, N, M, D]
        x = x.reshape(B * N, M, D) # [B * N, M, D]
        x = self.feature_attention(x) # [B * N, M, D]
        x = x.reshape(B, N, M, D)
        #each M token now has been cross attended with each other M token and possibly itself.

        x = x.permute(0, 2, 1, 3) # [B, M, N, D]

        x = x.reshape(B * M, N, D) # [B * M, N, D]
        x = self.patch_attention(x) # [B * M, N, D]
        x = x.reshape(B, M, N, D) # [B, M, N, D]
        #each N token now has been cross attended with each other N.

        return x
    
    
class patch_transformer(nn.Module):
    """ Patch-based Transformer encoder
    First normalizes the input with RevIN, splits each feature
    into overlapping patches, projects each patch to d_model
    adds feature and positional embeddings, 
    applies attention over features and patches sequentially 
    finally maps the encoded representation to the forecast horizon.
    
    Notes:
        patching idea: (https://arxiv.org/abs/2211.14730)
        sequential attention idea: (https://arxiv.org/html/2503.17658v1)

    Args
    ----------
    lookback_window : int
        number of past timesteps provided as input
    forecast_horizon : int
        number of future timesteps to predict.
    d_model : int
        embedding dimension used inside transformer blocks
    nhead : int
        num of attention heads in each transformer block
    dropout : float
        dropout probability
    num_features: int
        number of input channels
    num_blocks : int
        number of attention blocks (vertically stacked)
    
    Example
    ---------
    >>> model = patch_transformer(
    ...     lookback_window = 256, 
    ...     forecast_horizon = 4, 
    ...     d_model = 256, 
    ...     nhead = 8, 
    ...     dropout = 0.0, 
    ...     num_features = 17, 
    ...     num_blocks = 1
    ... )
    """

    def __init__(
        self, 
        lookback_window = 100, 
        forecast_horizon = 4, 
        d_model = 256, 
        nhead = 8, 
        dropout = 0.1, 
        num_features = 17, 
        num_blocks = 3
    ):
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

        self.blocks = nn.ModuleList([
            attention_module(
                lookback_window = lookback_window, 
                d_model = d_model, 
                nhead = nhead, 
                dropout = dropout) for i in range(num_blocks)]
            )
        
        self.linear_end = nn.Linear(n_patches * d_model, forecast_horizon, dtype=torch.float32)
        
    def forward(self, x):
        """ returns the predicted horizon from the input window

        Args
        ----------
        x : tensor (batch, seq_len, feature) -> (B, L, M)
            the input window to the model
        
        Returns
        ----------
        denormalized tensor (batch, horizon, feature) -> (B, horizon, M)

        Example
        ----------
        >>> x = torch.randn(32, 256, 17)
        >>> prediction = model(x)
        >>> prediction.shape
        torch.Size([32, 4, 17])
        """
        B, L, M = x.shape
        x = self.revin_layer(x, 'norm')

        x = self.patch(x)
        # x is [B, M, N, P]

        # each patch is projected in the space needed for first attention layer
        x = self.input_proj(x) #[B, M, N, P] @ [P, D] = [B, M, N, D] 
        x = x + self.feature_embed
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x) # [B, M, N, D] 
        
        x = x.flatten(2) # [B, M, N * D]
        x = self.linear_end(x) # [B, M, horizon]
        x = torch.transpose(x, 1, 2) # [B, horizon, M] to match input [B, L, M]
        return self.revin_layer(x, 'denorm')