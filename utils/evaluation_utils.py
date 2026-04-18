import numpy as np
import torch
from torch.utils.data import DataLoader
import utils.config as config

def compute_residuals(device, model, dataset, batch_size = 1024):
    """ runs the model in inference mode over the whole dataset and computes 
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
                        shuffle = False, num_workers = 2, pin_memory = True)
    residuals = []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            # [B, H * M] for forecasters, [B, L * M] for autoencoders
            residual = (y - y_pred).reshape(y.shape[0], -1) 
            residuals.append(residual) 

    # Concatenate all batches:
    # [[B, D], [B, D], ...] -> [N, D], where D is either H*M or L*M 
    residuals = torch.cat(residuals, dim=0)            
    return residuals

def nearest_neighbor_averaged_distance(R_test, R_train, R_train_norm, k=7):
    """ mean k-nearest squared residual distance to test tensor 
    (euclidean distance but not taking the root)

    For each test residual tensor, compute its distance to all training residuals
    and return the mean distance from the k nearest tensors.

    Args
    ---------
    R_test : tensor (B, D) -> (batch, flattened sample residual)
        normalized test/val residual sample to compare with the train residuals
    R_train : tensor (N, D) -> (all samples, flattened sample residua)
        normalized entire train residuals
    R_train_norm : tensor (1, N)
        unsqueezed precomputed R_train l2 norm
    k : int
        how many nearest neighbors to find
    
    Returns
    ---------
    score : tensor (B, )
        residual score for each sample in batch, 
        defined as the mean of the k nearest neighbors to R_test in R_train

    Example
    ---------
    >>> residual = torch.randn(2, 100)
    >>> train_residuals = torch.randn(32, 100)
    >>> R_train_norm = (train_residuals ** 2).sum(dim=1).unsqueeze(0)
    >>> distance = nearest_neighbor_averaged_distance(residual, train_residuals, R_train_norm)
    >>> distance.shape
    torch.Size([B])
    """
    B, D = R_test.shape 
    N = R_train_norm.shape[1]

    # broadcast explicitly for clarity
    R_test_norm = (R_test ** 2).sum(dim=1, keepdim = True).expand(B, N) # [B, 1] -> [B, N]
    R_train_norm = R_train_norm.expand(B, N) # [1, N] -> [B, N]

    # ||R_test - R_Train||^2 = ||R_test||^2 + ||R_Train||^2 - 2 * R_test @ R_Train.T
    dists = R_train_norm + R_test_norm - 2 * R_test @ R_train.T

    # get k nearest neighbors
    knn_dists, _ = torch.topk(dists, k, largest=False)

    # average distance
    return knn_dists.mean(dim=1)

def fit_custom_N2RE(device, model, train_dataset, batch_size = 1024):
    """ Pre computes metrics needed for Neighbor 2 REsiduals

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
    train_residuals : tensor (N, D)
        the normalized residual tensor
    train_mean : (D, )
        mean of flattened samples over trainig samples
    train_std : (D, )
        std deviation of flattened samples over trainig samples
    R_train_norm : tensor (1, N)
        the squared L2 norm of train_residuals, unsqueezed at dim 0 for broadcasting

    Example
    ---------
    >>> config.init("/path/to/data.csv", "/checkpoint_dir")
    >>> train_dataset = forecasting_Dataset(
    ...     'cpu', window_size=100, horizon=4, start=0, end=1000)
    >>> model = lstm_forecaster(hidden_size=128, horizon=4, num_layers=1)
    >>> train_residuals, train_mean, train_std, R_train_norm = fit_custom_N2RE(
    ...     'cpu', model, train_dataset
    ... )
    >>> train_residuals.shape
    torch.Size([N, D])
    """
    print("computing residuals")

    train_residuals = compute_residuals(device, model, train_dataset, batch_size)
    train_mean = train_residuals.mean(dim=0)
    train_std = train_residuals.std(dim=0) + 1e-8
    train_residuals = torch.tensor(
        (train_residuals - train_mean) / train_std, device=device, dtype=torch.float32)

    R_train_norm = (train_residuals ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
    return train_residuals, train_mean, train_std, R_train_norm

def score_custom_N2RE(
    device, 
    model,
    dataset, 
    train_residuals, 
    train_mean, 
    train_std, 
    R_train_norm, 
    batch_size = 1024
):
    """ minimal implementation of Neighbor 2 REsiduals computation
    original idea from (https://dl.acm.org/doi/10.1145/3477314.3506990) 

    Args
    ---------
    device : torch.device
        models device
    model : valid torch object of a class extending nn.Module
        current model (lstm_forecaster, lstm_ae, or patch_transformer)
    dataset : valid torch.utils.data.Dataset object
        current dataset
    train_residuals : tensor (N, D)
        the normalized residual tensor
    train_mean : (D, )
        mean of flattened samples over trainig samples
    train_std : (D, )
        std deviation of flattened samples over training samples
    R_train_norm : tensor (1, N)
        the squared L2 norm of train_residuals
    batch_size : int

    Returns
    ---------
    tuple[np.ndarray, np.ndarray, np.ndarray]: Scores, labels, and categories.

    Example
    ---------
    >>> config.init("/path/to/data.csv", root_dir + "/checkpoint_dir")
    >>> model = lstm_forecaster(hidden_size = 128, horizon = 4, num_layers = 1)
    >>> train_dataset = forecasting_Dataset(
    ...     'cpu', window_size=100, horizon=4, start=0, end=1000)
    >>> val_dataset = forecasting_Dataset(
    ...     'cpu', window_size=100, horizon=4, scaler = train_dataset.scaler, 
    ...     start=1000, end=2500, train = False)
    >>> stats = fit_custom_N2RE('cpu', model, train_dataset)
    >>> val_scores, val_labels, val_cats = \ 
    ...     score_custom_N2RE('cpu', model, val_dataset, *stats)
    >>> val_scores
    """
    print("Beginning Inference")
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    scores, labels, cats = [], [], []
    model.eval()
    with torch.inference_mode():
        for X, y, label, category in test_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            residual = (y - y_pred).reshape(y.shape[0], -1)  # [B, H * M] = [B, D]
            residual = (residual - train_mean) / train_std

            distance = nearest_neighbor_averaged_distance(
                residual, train_residuals, R_train_norm)

            scores.append(distance.cpu().numpy())
            labels.extend(label.numpy())
            cats.extend(category)

    scores = np.concatenate(scores)
    return scores, np.array(labels), np.array(cats)

