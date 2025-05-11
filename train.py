# training.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

def train(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    running_loss = 0.0
    early_stop_counter = 0
    early_stop_threshold = 0.10
    patience = 5

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if target.size() != torch.Size([1, 14, 264, 496]):
            print("target", target.size())
            continue
        optimizer.zero_grad()
        reconstructed_image = model(data)
        loss = loss_fn(reconstructed_image, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")        
        if loss.item() < early_stop_threshold:
            early_stop_counter += 1
        else:
            early_stop_counter = 0
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}, batch {batch_idx}")
            break  

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch}] Average Loss: {avg_loss}")
    return avg_loss

def val(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            if target.size() != torch.Size([1, 14, 264, 496]):
                print("target", target.size())
                continue
            reconstructed_image = model(data)
            loss = loss_fn(reconstructed_image, target)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Val Average Loss: {avg_loss}")
    return avg_loss

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found. Starting from epoch 0.")
        return 0