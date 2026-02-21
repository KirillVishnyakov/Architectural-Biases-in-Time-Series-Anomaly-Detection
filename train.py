import numpy as np
import torch.nn as nn
import torch


def train_lstm(model, exp_name, train_dataset, test_dataset, lr, batch_size, num_epochs, ):
    num_batches = len(train_dataset) // batch_size
    train_mse_array = np.zeros(num_epochs)
    test_mse_array = np.zeros(num_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

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
            print(f"experiment: {exp_name}. epoch {epoch}, train: MSE {train_mse_array[epoch]:.4f} test MSE: {test_mse_array[epoch]:.4f}")
    
    return model, train_mse_array, test_mse_array