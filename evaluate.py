
import torch
def evaluate_lstm_threshold(model, treshold, forecasting_dataset):
    TP, FP = 0, 0
    for idx in range(len(forecasting_dataset)):
        X, Y, label = forecasting_dataset[idx]
        y_pred = model(X.unsqueeze(dim = 0))
        error = torch.linalg.norm(Y - y_pred.squeeze(0)).item()

        if error > treshold and label == 1: 
            TP+=1 
        elif error > treshold and label == 0:
            FP+=1

    if TP == 0 or FP == 0: return 0, 0, 0
    recall = TP/forecasting_dataset.total_anomalies
    precision = TP/(TP + FP)
    print(f"| evaluation_treshold: {treshold} | recall {recall}, precision {precision}, F1 {2 * (precision * recall)/(precision + recall)}")
    return precision, recall, 2 * (precision * recall)/(precision + recall)