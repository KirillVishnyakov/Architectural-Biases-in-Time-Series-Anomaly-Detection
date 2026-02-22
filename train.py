import numpy as np
import torch.nn as nn
import torch
import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

class LrPlateauScheduler:
    def __init__(self, patience=3, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.min_delta = min_delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_lstm(model, exp_name, train_dataset, test_dataset, lr, batch_size, num_epochs):
    num_batches = len(train_dataset) // batch_size
    train_mse_array = np.zeros(num_epochs)
    test_mse_array = np.zeros(num_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    earlyStopper = EarlyStopping()
    LrPlateauSchedule = LrPlateauScheduler()

    for epoch in range(num_epochs):
        model.train()
        for batch_idx in range(num_batches):
            window_batch = train_dataset[batch_idx * batch_size: (batch_idx + 1)*batch_size]
            y_pred_batch = model(window_batch[0]).unsqueeze(dim = 1)
            loss = loss_fn(window_batch[1].unsqueeze(dim = 1), y_pred_batch)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  
        with torch.no_grad():
            
            train_mse_array[epoch] = loss_fn(model(train_dataset.X), train_dataset.y).item()
            test_mse_array[epoch] = loss_fn(model(test_dataset.X), test_dataset.y).item()

            print(f"| experiment: {exp_name} | epoch {epoch}, train: MSE {train_mse_array[epoch]:.4f}, test MSE: {test_mse_array[epoch]:.4f}")
            if LrPlateauSchedule(test_mse_array[epoch]):
                current_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = current_lr * 0.5
                print(f"update LR: {current_lr} -> {optimizer.param_groups[0]['lr']}")
            if earlyStopper(test_mse_array[epoch]):
                print("Stopping early")
                break
            if test_mse_array[epoch] < earlyStopper.best_score + earlyStopper.min_delta:
                best_model_wts = copy.deepcopy(model.state_dict())
    return best_model_wts, train_mse_array[:epoch], test_mse_array[:epoch]