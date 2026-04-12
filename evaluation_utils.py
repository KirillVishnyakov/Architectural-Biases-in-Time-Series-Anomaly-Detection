import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
import utils.config as config

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