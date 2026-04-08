import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import deque
import numpy as np
import config as config

def robust_scaled_feature_mse(feature_buffer, feat_err):
    buffer_arr = np.array(feature_buffer)
    # Compute median for each feature seen recentyl
    med = np.median(buffer_arr, axis=0)
    mad = np.median(np.abs(buffer_arr - med), axis=0)
    scale = np.clip(mad, 1e-8, None)

    return np.mean(((feat_err - med) / scale) ** 2, axis=1)


def calculate_scores(device, model, dataset, batch_size = 1024, window_size = 200, min_history = 50, q = 0.99):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores, labels, categories = [], [], []
    flags = [], []
    
    # sliding window keeping track of past residuals
    feature_buffer = deque(maxlen=window_size) # deque for popping front in O(1)
    threshold_buffer = deque(maxlen=window_size)
    
    model.eval()
    with torch.no_grad():
        for X, y, label, category in loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            residuals = (y - y_pred).cpu().numpy()  # (batch, horizon, features)
            feature_error = np.mean(residuals**2, axis = 1) # (batch, features) 
            
            batch_scores = \
            robust_scaled_feature_mse(feature_buffer, feature_error) if len(feature_buffer) >= min_history else np.full(len(feature_error), np.nan) # better than np.zeros(len(feature_error))

            # determine if each score is anomalous or not based on rolling threshold
            for s_t in batch_scores:
                #batch_scores are filled with nan when len(feature_buffer) < min_history
                if np.isnan(s_t):
                    flags.append(False)
                    continue
                recent = np.array(threshold_buffer)
                flag = s_t > np.quantile(recent, q) if len(threshold_buffer) >= min_history else False
                flags.append(flag)
                threshold_buffer.append(s_t)
                 
            #will pop in O(1) the beginning if length exceeds window size
            feature_buffer.extend(feature_error)

            scores.extend(batch_scores)
            labels.extend(label.numpy())
            categories.extend(category)
    
    return {"scores": np.array(scores), "labels": np.array(labels), "cats": np.array(categories), "flags": np.array(flags)}