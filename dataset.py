import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import RobustScaler
import config as config


# Stride of 1 is implicitly implied. Window size can be set.
class forecasting_Dataset(data.Dataset):
    def __init__(self, device, window_size, horizon = 1, scaler = None, start = 0, end = 90000, train = True):
        self.horizon = horizon
        self.train = train
        self.window_size = window_size
        self.device = device


        #log transform has to be applied before scaler, or the scaler will get thrown off by outliers (in log_transform_cols)
        if self.train:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["y", "category", "timestamp"], axis = 1)
            self.scaler = RobustScaler().fit(self.dataset)
            
        else:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["timestamp"], axis = 1)
            self.labels = self.dataset["y"].values
            self.categories = self.dataset["category"].values
            self.total_anomalies = np.sum(self.labels == 1)
            self.dataset = self.dataset.drop(["y", "category"], axis = 1)
            self.scaler = scaler
        #clip doesnt care about outliers, after scaler is fine
        normalized = np.clip(self.scaler.transform(self.dataset), -5, 5)
        self.normalized_dataset = torch.from_numpy(normalized).float()

    def __len__(self):
        return len(self.normalized_dataset) - self.window_size - self.horizon

    def __getitem__(self, idx):
        X = self.normalized_dataset[idx : idx + self.window_size]
        y = self.normalized_dataset[idx + self.window_size: idx + self.window_size + self.horizon]

        if self.train:
            return X, y
        else: 
            label = int(np.any(self.labels[idx + self.window_size: idx + self.window_size + self.horizon])) # if any label is non zero, treat the prediction as anomalous
            category = self.categories[idx + self.window_size] # just take the first category since anomalies are contiguous
            return X, y, label, category