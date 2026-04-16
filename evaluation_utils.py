import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import utils.config as config

def compute_residuals(device, model, dataset, batch_size = 1024):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=False, num_workers = 2, pin_memory = True)
    residuals = []
    model.eval()
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            residual = (y - y_pred).reshape(y.shape[0], -1).cpu().numpy() # [B, H * M]
            residuals.append(residual) 
    """
    [ residuals
        [B, H * M] batch 1
        [B, H * M] batch 2
    ]
    """
    residuals = np.concatenate(residuals, axis=0) # [list of [B, H * M]] -> [stacked [N, H * M]]
    return residuals

def fit_knn(train_residuals, k):
    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1
    )
    nn.fit(train_residuals)
    return nn

def nearest_neighbor_averaged_distance(R_test, nn):
    dists, _ = nn.kneighbors(R_test)   # [B, k]
    return (dists ** 2).mean(axis=1)

def custom_N2RE(device, model, train_dataset, val_dataset, batch_size = 1024):
    print("computing residuals")
    train_residuals = compute_residuals(device, model, train_dataset, batch_size)
    train_mean = np.mean(train_residuals, axis=0)
    train_std  = np.std(train_residuals, axis=0) + 1e-8
    train_residuals = (train_residuals - train_mean ) / train_std
    print("Fitting knn")
    nn = fit_knn(train_residuals, k = 5)
    test_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers = 2, pin_memory = True)
    scores, labels, cats = [], [], []
    model.eval()
    print("Beginning Inference")
    with torch.inference_mode():
        for X, y, label, category in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            residual = (y - y_pred).reshape(y.shape[0], -1).cpu().numpy() # [B, H * M]
            residual = (residual - train_mean) / train_std

            distance = nearest_neighbor_averaged_distance(residual, nn)
            scores.append(distance)

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