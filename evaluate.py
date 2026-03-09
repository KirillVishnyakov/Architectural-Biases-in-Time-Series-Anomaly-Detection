
import torch
import numpy as np
from torch.utils.data import DataLoader

def evaluate_lstm_scores(model, forecasting_dataset):
    loader = DataLoader(forecasting_dataset, batch_size=64, shuffle=False)
    scores, labels, categories = [], [], []
    with torch.no_grad():
        for X, Y, label, category in loader:
            y_pred = model(X)
            error = torch.linalg.norm(Y - y_pred, dim=-1) 
            scores.extend(error.cpu().numpy())
            labels.extend(label.cpu().numpy())
            categories.extend(category)
    return np.array(scores), np.array(labels), np.array(categories)