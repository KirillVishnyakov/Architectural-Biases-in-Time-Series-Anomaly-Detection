import numpy as np
import torch
from torch.utils.data import DataLoader
import utils.config as config

def compute_residuals(device, model, dataset, batch_size = 1024):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = 2, pin_memory = True)
    residuals = []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            residual = (y - y_pred).reshape(y.shape[0], -1) # [B, H * M]
            residuals.append(residual) 
    """
    [ residuals
        [B, H * M] batch 1
        [B, H * M] batch 2
    ]
    """
    residuals = torch.cat(residuals, dim=0) # [list of [B, H * M]] -> [stacked [N, H * M]]
    return residuals
def nearest_neighbor_averaged_distance(R_test, R_train, R_train_norm, k=7):
    # R_test: [B, D]
    B, D = R_test.shape
    # R_train_norm: [1, N]
    N = R_train_norm.shape[1]
    # R_train: [N, D]
    """
    computed as ||R_test - R_Train||
    simplify: ||R_test - R_Train|| = ||R_test|| + ||R_Train|| - 2 * R_test @ R_Train.T
    basically only need to compute ||R_Train|| once
    """
    R_test_norm = (R_test ** 2).sum(dim=1, keepdim = True).expand(B, N) # [B, 1] -> [B, N]
    R_train_norm = R_train_norm.expand(B, N) # [1, N] -> [B, N]
    dists = R_train_norm + R_test_norm - 2 * R_test @ R_train.T

    # get k nearest neighbors
    knn_dists, _ = torch.topk(dists, k, largest=False)

    # average distance
    return knn_dists.mean(dim=1)

def fit_custom_N2RE(device, model, train_dataset, batch_size = 1024):
    print("computing residuals")

    train_residuals = compute_residuals(device, model, train_dataset, batch_size)
    train_mean = train_residuals.mean(dim=0)
    train_std = train_residuals.std(dim=0) + 1e-8
    train_residuals = torch.tensor((train_residuals - train_mean) / train_std, device=device, dtype=torch.float32)

    R_train_norm = (train_residuals ** 2).sum(dim=1).unsqueeze(0)  # [1, N]
    return train_residuals, train_mean, train_std, R_train_norm

def score_custom_N2RE(device, model, dataset, train_residuals, train_mean, train_std, R_train_norm, batch_size = 1024):
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

            distance = nearest_neighbor_averaged_distance(residual, train_residuals, R_train_norm)

            scores.append(distance.cpu().numpy())
            labels.extend(label.numpy())
            cats.extend(category)

    scores = np.concatenate(scores)
    return scores, np.array(labels), np.array(cats)


"""
def calculate_gauss_distribution(device, model, dataset, batch_size = 1024):
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = 2, pin_memory = True)
    model.eval()
    error_vectors = []
    with torch.inference_mode():
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            error = (y - y_pred).reshape(len(X), -1) #(timesteps, l*17)
            error_vectors.append(error.cpu().numpy())

    E = np.concatenate(error_vectors, axis=0) #[(batch 1, l*17), (batch 2, l*17), ...] -> (all train timesteps, l * 17)
    mu = E.mean(axis=0)
    sigma = np.diag(E.var(axis=0) + 1e-6)

    dist = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return dist



def evaluate_lstm_scores(device, model, dataset, dist, batch_size = 1024):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = 2, pin_memory = True)
    scores, labels, categories = [], [], []
    model.eval()
    with torch.inference_mode():
        for X, y, label, category in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            error = (y - y_pred).reshape(len(X), -1).cpu().numpy()
            score = -dist.logpdf(error)  #high = anomaly
            scores.extend(score)
            labels.extend(label.numpy())
            categories.extend(category)
    return np.array(scores), np.array(labels), np.array(categories)


def fit_threshold(device, model, train_dataset, val_dataset, batch_size = 1024):
    train_residuals = compute_residuals(device, model, train_dataset, batch_size)
    step = 500000
    d2 = []

    for val_start in range(step, len(train_residuals), step):
        train_window = train_residuals[: val_start]
        val_window = train_residuals[val_start : val_start + step]
        if len(val_window) == 0:
            break
        mcd = MinCovDet().fit(train_window)
        d2.extend(mcd.mahalanobis(val_window))
    d2 = np.asarray(d2)
    clean_mask = d2 < np.percentile(d2, 95)
    t = np.percentile(d2[clean_mask], 99)

    mcd_final = MinCovDet().fit(train_residuals)
    del train_residuals
    
    val_residuals = compute_residuals(device, model, val_dataset, batch_size)
    d2_inference = mcd_final.mahalanobis(val_residuals)

    anomalies = d2_inference > t
    del val_residuals
    return anomalies, t

"""