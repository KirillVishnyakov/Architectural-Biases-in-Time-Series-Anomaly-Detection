import torch.nn as nn
import torch
import copy
from torch.utils.data import DataLoader

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
                self.counter = 0
                return True
        return False

def initialize_weights_xavier(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

def fit_lstm(device, model, exp_name, train_dataset, test_dataset, lr, batch_size, num_epochs):
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 1, pin_memory = True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size = 1, num_workers = 1, pin_memory = True, persistent_workers=True)
    
    model.apply(initialize_weights_xavier)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    earlyStopper = EarlyStopping()
    LrPlateauSchedule = LrPlateauScheduler()

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred_batch = model(X)
            loss = loss_fn(y, y_pred_batch)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        eval_losses = []
        model.eval()
        with torch.no_grad():
            for X, y, _, __ in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred_batch = model(X)
                loss = loss_fn(y, y_pred_batch)
                eval_losses.append(loss.item())
            if True:
                print(f"|{exp_name}| train = {sum(train_losses)/len(train_losses):.4f}, test= {sum(eval_losses)/len(eval_losses):.4f}")
            avg_eval_loss = sum(eval_losses) / len(eval_losses)
            avg_train_loss = sum(train_losses) / len(train_losses)

            if avg_eval_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = avg_eval_loss

            if LrPlateauSchedule(avg_eval_loss):
                current_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = current_lr * 0.5
                print(f"update LR: {current_lr} -> {optimizer.param_groups[0]['lr']}")
            if earlyStopper(avg_eval_loss):
                print(f"| experiment: {exp_name} | epoch {epoch + 1}, train: MSE {avg_train_loss:.4f}, test MSE: {avg_eval_loss:.4f}")
                print("Stopping early")
                break

    return best_model_wts, best_loss