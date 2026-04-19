import os
import torch.nn as nn
import torch
import copy
import math
from torch.utils.data import DataLoader
import utils.config as config

class EarlyStopping:
    """ implements early stopping module
    
    Args
    ---------
    patience : int 
        early stopping patience (in epochs)
    min_delta : float 
        minimum improvement to reset early stopping
    mode (optional)

    Example
    ---------
    >>> earlyStopper = EarlyStopping(patience = 10, min_delta = 0.001)
    """

    def __init__(self, patience=15, min_delta=0.00005, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """ early stopper call and counter

        Args
        --------
        score : float
            loss at this step
        
        Returns
        ---------
        True if score hasnt improved over patience epochs
        False otherwise and resets counter
            
        Example
        ---------
        >>> if earlyStopper(avg_eval_loss) : break
        # see in fit()
        """
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

class WarmupCosineScheduler:
    """ implements lr scheduler module: linear warmup then cosine decay

    Args
    ---------
    optimizer : torch.optim
        the optimizer object (ex: adamW)
    warmup_steps : int
        training steps until reaching max lr (linear landscape)
    total_steps : int
        total training steps
    min_lr : float
        minimum value the lr can decay to
    
    Example
    ---------
    >>> scheduler = WarmupCosineScheduler(
    ...    optimizer, 
    ...    warmup_steps=warmup_steps,
    ...    total_steps=total_steps,
    ...    min_lr=lr * 0.01
    ... ) # see in fit()
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        """ updates model/optimizer learning rates according to stored lr.
        Example
        --------
        >>> scheduler.step()
        """
        self.current_step += 1
        lr = self.get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        """ computes the learning rate according to trainings progress in the linear/cosine landscape

        Returns
        --------
            float: Learning rate at the current step.
        """
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.current_step / max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def state_dict(self):
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']


def transform_data(device, data, mode = 'train'):
    """ apply white noise to input when training

    Example
    --------
    >>> x = torch.randn(2, 32, 100)
    >>> x_transformed = transform_data('cpu', x, "train")
    >>> x_transformed.shape
    """
    if mode == 'train':
        #white noise
        noisy_data = data + torch.randn_like(data, device = device) * 0.05
    return noisy_data


def init_weights_xavier(module):
    """ Applies xaver initialization to linear layer and lstm layers

    Example
    --------
    >>> model = lstm_forecaster(hidden_size = 128, horizon = 4, num_layers = 1)
    >>> model.apply(init_weights_xavier)
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param.data)


def fit(device, model, exp_name, train_dataset, test_dataset, 
                   lr, batch_size, num_epochs, shuffle = False, 
                   patience = 15, min_delta = 0.00005):
    """ Train a model with early stopping and cosine LR scheduling.
    Saves checkpoints.

    Args
    --------
    device (torch.device): device used for training
    model (nn.Module): forecasting model to train
    exp_name (str): experiment name (logging)
    train_dataset (torch.utils.data.Dataset): training dataset
    test_dataset (torch.utils.data.Dataset): validation/test dataset
    lr (float): learning rate
    batch_size (int): training batch size
    num_epochs (int): maximum number of training epochs
    shuffle (bool): whether to shuffle training data
    patience (int): early stopping patience (in epochs)
    min_delta (float): minimum improvement to reset early stopping

    Returns
    --------
    tuple (best_model_wts, best_loss, avg_train_loss)

    Example
    --------
    >>> config.init("/path/to/data.csv", root_dir + "/checkpoint_dir")
    >>> model = lstm_forecaster(hidden_size = 128, horizon = 4, num_layers = 1)
    >>> train_dataset = forecasting_Dataset(
    ...     'cpu', window_size=100, horizon=4, start=0, end=1000)
    >>> val_dataset = forecasting_Dataset(
    ...     'cpu', window_size=100, horizon=4, scaler = train_dataset.scaler, 
    ...     start=1000, end=2500, train = False)
    >>> weights, val_loss, train_loss = fit(
    ...      'cpu', model, "exp", train_dataset, val_dataset,
    ...     lr=1e-3, batch_size=32, num_epochs=10
    ... )
    >>> train_loss
    """
    model.apply(init_weights_xavier)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 1, pin_memory = True, persistent_workers=True, shuffle = shuffle)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 1, pin_memory = True, persistent_workers=True)

    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    # according to GPT-2 architecture and onwards, you should remove weight decay from bias's and layer norms
    optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() 
                if p.requires_grad and p.dim() >= 2], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() 
                if p.requires_grad and p.dim() < 2], 'weight_decay': 0.0}
    ], lr=lr)
    
    loss_fn = nn.MSELoss()
    earlyStopper = EarlyStopping(patience = patience, min_delta = min_delta)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) // 2 #
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=lr * 0.01
    )
    print(f"LR Scheduler: {warmup_steps} warmup steps, {total_steps} total steps")

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        if hasattr(model, 'decoder'):
            model.decoder.current_epoch = epoch
            model.decoder.total_epochs = num_epochs
        train_losses = []
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            X = transform_data(device, X, "train")
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_pred_batch = model(X)
                loss = loss_fn(y, y_pred_batch)

            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        eval_losses = []
        model.eval()
        with torch.no_grad():
            for X, y, _, __ in test_loader:
                X, y = X.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y_pred_batch = model(X)
                    loss = loss_fn(y, y_pred_batch)
                eval_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_eval_loss = sum(eval_losses) / len(eval_losses)

            if (epoch+1) % 1 == 0:
                print(f"|{exp_name}| train = {avg_train_loss:.4f} | test= {avg_eval_loss:.4f} | LR: {scheduler.get_lr():.2e}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, f"{checkpoint_dir}/epoch_{epoch+1}.pt")

            
            if avg_eval_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = avg_eval_loss

            if earlyStopper(avg_eval_loss):
                print(f"| experiment: {exp_name} | epoch {epoch + 1}, train: MSE {avg_train_loss:.4f}, test MSE: {avg_eval_loss:.4f}")
                print("Stopping early")
                break
            
    return best_model_wts, best_loss, avg_train_loss