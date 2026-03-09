
import torch
import numpy as np

def evaluate_lstm_threshold(model, threshold, forecasting_dataset):
    scores = []
    labels = []

    for idx in range(len(forecasting_dataset)):
        X, Y, label = forecasting_dataset[idx]
        y_pred = model(X.unsqueeze(dim = 0))
        error = torch.linalg.norm(Y - y_pred.squeeze(0)).item()
        scores.append(error)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    TP = np.sum((scores > threshold) & (labels == 1))
    FP = np.sum((scores > threshold) & (labels == 0))

    if TP == 0 or FP == 0: return 0, 0, 0

    recall = TP/forecasting_dataset.total_anomalies
    precision = TP/(TP + FP)

    print(f"| evaluation_threshold: {threshold} | recall {recall}, precision {precision}, F1 {2 * (precision * recall)/(precision + recall)}")

    return scores, labels, precision, recall, 2 * (precision * recall)/(precision + recall)