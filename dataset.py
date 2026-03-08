import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import RobustScaler


# Stride of 1 is implicitly implied. Window size can be set.
class LSTM_Dataset(data.Dataset):
    def __init__(self, device, window_size, start = 0, end = 90000, train = True):
        self.train = train
        if self.train:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["y", "category", "timestamp"], axis = 1)
        else:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["timestamp"], axis = 1)
            self.labels = self.dataset["y"].values
            self.total_anomalies = np.sum(self.labels == 1)
            self.dataset = self.dataset.drop(["y", "category"], axis = 1)

        self.normalized_dataset = torch.from_numpy(RobustScaler().fit_transform(self.dataset))

        self.X, self.y = torch.zeros((end - start - window_size, window_size, 17)), torch.zeros((end - start - window_size, 17))

        for i in range(len(self.normalized_dataset) - window_size):
            window = self.normalized_dataset[i: i+window_size]
            target = self.normalized_dataset[i+window_size]

            self.X[i] = window
            self.y[i] = target
        self.X = self.X.to(device)
        self.y = self.y.to(device)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx], self.y[idx]
        else: 
            label = self.labels[idx + self.window_size]
            return self.X[idx], self.y[idx], label


