import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import RobustScaler


# Stride of 1 is implicitly implied. Window size can be set.
class LSTM_tuning_DataSet(data.Dataset):
    def __init__(self, device, window_size, start = 0, end = 90000):
        self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["y", "category", "timestamp"], axis = 1)
        self.normalized_dataset = RobustScaler().fit_transform(self.dataset)

        
        self.clean_data =  torch.from_numpy(self.normalized_dataset[start: end])
        
        self.X, self.y = torch.zeros((end - start - window_size, window_size, 17)), torch.zeros((end - start - window_size, 17))
        for i in range(len(self.clean_data) - window_size):
            window = self.clean_data[i: i+window_size]
            target = self.clean_data[i+window_size]

            self.X[i] = window
            self.y[i] = target
        self.X = self.X.to(device)
        self.y = self.y.to(device)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
