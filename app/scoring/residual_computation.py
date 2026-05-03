import numpy as np
import torch
from torch.utils.data import DataLoader
import app.utils.config as config

def compute_residuals(device, model, dataset, batch_size = 1024, train = True):
    """ runs the model in inference mode over a dataset and computes 
    difference between true and predicted features, flattened per sample.

    Args
    ---------
    device : torch.device
        models device
    model : valid torch object of a class extending nn.Module
        current model (lstm_forecaster, lstm_ae, or patch_transformer)
    dataset : valid torch.utils.data.Dataset object
        current dataset
    batch_size : int

    Returns
    ---------
    tensor (N = windows, D = horizon * feature) for forecasters
    tensor (N = windows, D = seq_length * feature) for autoencoders
        the residual tensor
    
    Example:
    ---------
    >>> config.init("/path/to/data.csv", "/checkpoint_dir")
    >>> train_dataset = forecasting_Dataset('cpu', window_size=100, horizon=4, start=0, end=1000)
    >>> model = lstm_forecaster(hidden_size=128, horizon=4, num_layers=1)
    >>> train_residuals = compute_residuals(
    ...     'cpu', model, train_dataset
    ... )
    >>> train_residuals.shape
    torch.Size([N, D])
    # same as [end - start - window_size - horizon, D]
    """
    loader = DataLoader(dataset, batch_size = batch_size, 
                        shuffle = False, num_workers = 1, pin_memory = False)
    residuals, labels, cats = [], [], []
    model.eval()
    with torch.inference_mode():
        if train:
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                # [B, H * M] for forecasters, [B, L * M] for autoencoders
                residual = (y - y_pred).reshape(y.shape[0], -1) 
                residuals.append(residual) 
        else:
            for X, y, label, cat in loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                residual = (y - y_pred).reshape(y.shape[0], -1) 
                residuals.append(residual) 
                labels.append(label)
                cats.append(cat)
        # Concatenate all batches:
        # [[B, D], [B, D], ...] -> [N, D], where D is either H*M or L*M 
        residuals = torch.cat(residuals, dim=0)      
        labels = torch.cat(labels, dim=0)   
        cats = torch.cat(cats, dim=0)   
    return residuals, labels, cats
