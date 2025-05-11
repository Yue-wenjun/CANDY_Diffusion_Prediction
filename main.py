# main.py
import torch
from data_loading import get_data_loaders
from models.diffusion import DiffusionModel
from train import train, val, save_checkpoint, load_checkpoint
from utils import app
import os
import time

# Hyperparameters
batch_size = 1
in_channel = 14
hidden_channel = 14
out_channel = 14
input_size = 496
hidden_size = 264
T = 20
learning_rate = 0.0005
num_epochs = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
samples_file = "..."
window_size = 14
stride = 1
train_loader, val_loader, test_loader = get_data_loaders(
    samples_file, window_size, stride, batch_size, device
)

# Model, loss, optimizer
model = DiffusionModel(
    batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size, T
).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoint
checkpoint_path = "checkpoint.pth"
start_epoch = 0
if os.path.exists(checkpoint_path):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
else:
    print(
        f"No checkpoint found at {checkpoint_path}, starting from epoch {start_epoch}"
    )

# Training loop
epoch_times = []
for epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()  
    train_loss = train(model, train_loader, optimizer, criterion, device, epoch)   
    val_loss = val(model, val_loader, criterion, device)
    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_times.append(epoch_time)
    save_checkpoint(model, optimizer, epoch, checkpoint_path)
    print(
        f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Epoch Time: {epoch_time:.2f} seconds"
    )
total_training_time = sum(epoch_times)
print(f"Total Training Time: {total_training_time:.2f} seconds") 

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

test = app(model, test_loader, device, criterion)
