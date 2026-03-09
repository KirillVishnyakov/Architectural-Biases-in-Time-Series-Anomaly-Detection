
import torch
import numpy as np

def evaluate_lstm_scores(model, forecasting_dataset):
    scores = []
    labels = []
    with torch.no_grad():
        for idx in range(len(forecasting_dataset)):
            X, Y, label = forecasting_dataset[idx]
            y_pred = model(X.unsqueeze(dim = 0))
            error = torch.linalg.norm(Y - y_pred.squeeze(0)).item()
            scores.append(error)
            labels.append(label)

    return np.array(scores), np.array(labels)