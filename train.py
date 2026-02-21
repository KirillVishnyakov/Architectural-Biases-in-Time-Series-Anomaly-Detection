import numpy as np
import torch.nn as nn
import torch


def train_lstm(model, train_dataset, test_dataset, optimizer, batch_size, num_epochs, loss_fn):
    num_batches = len(train_dataset) // batch_size
    train_mse_array = np.zeros(num_epochs)
    test_mse_array = np.zeros(num_epochs)

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
            print("epoch %d, train: MSE %.4f test MSE: %.4f" % (epoch, train_mse_array[epoch], test_mse_array[epoch]))