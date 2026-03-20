import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader

def calculate_gauss_distribution(device, model, train_loader):
    model.eval()
    error_vectors = []
    with torch.no_grad():
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            error = (y - y_pred).view(len(X), -1) #(timesteps, l*17)
            error_vectors.append(error.cpu().numpy())

    E = np.concatenate(error_vectors, axis=0) #[(batch 1, l*17), (batch 2, l*17), ...] -> (all train timesteps, l * 17)
    mu = E.mean(axis=0)
    sigma = np.cov(E.T)

    dist = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return dist



def evaluate_lstm_scores(device, model, dataset, dist):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    scores, labels, categories = [], [], []
    model.eval()
    with torch.no_grad():
        for X, y, label, category in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            error = (y - y_pred).view(len(X), -1).cpu().numpy()
            score = -dist.logpdf(error)  #high = anomaly
            scores.extend(score)
            labels.extend(label.numpy())
            categories.extend(category)
    return np.array(scores), np.array(labels), np.array(categories)