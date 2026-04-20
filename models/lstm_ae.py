import torch.nn as nn
from utils.RevIN import RevIN
import torch
import random

class lstm_encoder(nn.Module):
    """ Implements the encoder module of the lstm autoencoder
    
    Args
    ---------
    num_features : int
        num of features
    lookback_window: int
        size of the input window the model sees (contiguous)
    latent_dim: int
        latent decoder dimension
    
    Example
    ---------
    >>> model = lstm_ae(17, lookback_window = 256, latent_dim = 5)
    >>> model_encoder = model.encoder
    """
    
    def __init__(self, num_features = 17, lookback_window = 50, latent_dim = 17):
        super(lstm_encoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(self.num_features, self.latent_dim, num_layers = 1, batch_first = True)
    
    def forward(self, x):
        """ Encode an input sequence with an LSTM

        Args
        ---------
        x: tensor (Batch, seq_len, features) -> (B, L, M)
            input tensor to encode

        Returns
        ---------
        tuple:
            out (torch.Tensor): LSTM outputs (B, L, latent_dim)
            (h, c): Final hidden and cell states (1, B, latent_dim)
            x (torch.Tensor): Input tensor
        
        Example
        ---------
        >>> x = torch.randn(32, 256, 17)
        >>> out, (h, c), x_out = model_encoder(x)
        >>> out.shape
        torch.Size([32, 256, latent_dim])
        """
        if x.dim() < 3: x = x.unsqueeze(0) # unsqueeze incase batch dim missing
        out, (final_hidden_State, final_cell_state) = self.lstm(x)

        return out, (final_hidden_State, final_cell_state), x 

class lstm_decoder(nn.Module):
    """ Implements the decoder module of the lstm autoencoder
    
    Args
    ---------
    latent_dim: int
        the latent decoder dimension
    lookback_window: int
        size of the input window the model sees (contiguous)
    num_features : int
        num of features in the lstm_autoencoder input, 
        the decoder reconstructs into that dimension 
    
    Example
    ---------
    >>> model = lstm_ae(17, lookback_window=256, embedding_dim_ratio=3.0)
    >>> model_decoder = model.decoder
    """

    def __init__(self, latent_dim, lookback_window = 50, num_features = 17):
        super(lstm_decoder, self).__init__()
        self.num_features = num_features
        self.lookback_window = lookback_window
        self.latent_dim = latent_dim
        self.total_epochs = 1 # init to 1 to avoid div by zero bug during inferene
        self.current_epoch = 0

        self.lstm = nn.LSTM(self.num_features + self.latent_dim, self.latent_dim, num_layers = 1, batch_first = True)
        self.linear = nn.Linear(self.latent_dim, self.num_features)

    def forward(self, x):  
        """ Returns the reconstructed input
        Args
        ---------
        x: Tuple of
            in_sequence: tensor (B, L, latent_dim)
                the encoded tensor
            h0: tensor (1, B, latent_dim)
                decoders initial hidden state (encoders final hidden state)
            c0: tensor (1, B, latent_dim)
                decoders initial cell state (encoders final cell state)
            x: tensor (B, L, M)
                the "answer" of what the decoder is trying to reconstruct, 
                used for teacher forcing during training
        
        Returns
        ---------
        reconstructed_window: tensor (B, L, M)
            the reconstructed input
        
        Example
        ---------
        >>> x = torch.randn(32, 256, 17)
        >>> reconstructed_window = model_decoder(model.encoder(x))
        >>> reconstructed_window.shape
        torch.Size([32, 256, 17])
        """
        in_sequence, (h0, c0), true = x
        B, L, H = in_sequence.shape
        teacher_forcing_prob = max(0.0, 1.0 - (self.current_epoch / self.total_epochs))
        
        if self.training and (teacher_forcing_prob > random.random()):
            # true_sequences = [X, ..., L-1] are the true timesteps (used for teacher forcing)
            true_sequences = true.roll(shifts = 1, dims = 1)
            # replace X by zeros, start token
            true_sequences[:, 0:1, :] = torch.zeros((B, 1, self.num_features), device = in_sequence.device, dtype = in_sequence.dtype)

            """
            in_sequence is what the encoder output at each time step [B, t, latent_dim]
            stacking [B, L, latent_dim] with [0, ..., L-1] = [B, L, latent_dim, 0, ..., L-1]
            means decoder input has information about previous true timestep 
            and encoders representation of current time step
            """
            decoder_input = torch.cat([true_sequences, in_sequence], dim = 2)

            reconstructed_window, _ = self.lstm(decoder_input, (h0, c0))
            reconstructed_window = self.linear(reconstructed_window)

        # when doing inference, no access to true_sequences, therefore decoder input is cat[last_decoder_output, in_sequence@i]
        # which cant be vectorized since we dont know decoder outputs in advance.
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
    """ This class implements the lstm auto_encoder

    Args
    ---------
    num_features : int
        num of features
    lookback_window: int
        size of the input window the model sees (contiguous)
    embedding_dim_ratio: float
        % of the input (num_features) dimension to keep or expand into the latent dimension

    Example
    ---------
    >>> model = lstm_ae(17, lookback_window=256, embedding_dim_ratio=3.0)
    """

    def __init__(self, num_features = 17, lookback_window = 50, embedding_dim_ratio = 0.30):
        super(lstm_ae, self).__init__()
        self.latent_dim = int(num_features * embedding_dim_ratio)
        self.encoder = lstm_encoder(num_features, lookback_window, self.latent_dim)
        self.revin_layer = RevIN(num_features)
        self.decoder = lstm_decoder(self.latent_dim, lookback_window, num_features)
    
    def forward(self, x):
        """ returns the output of the lstm autoencoder

        Args
        ---------
        x: tensor (Batch, seq_len, features) -> (B, L, M)
            input tensor to reconstruct
        
        Returns 
        ---------
        x: tensor (B, L, M)
            reconstructed input
        Example
        ---------
        >>> x = torch.randn(32, 256, 17)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 256, 17])
        """
        x = self.revin_layer(x, 'norm')
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.revin_layer(x, 'denorm')
        return x