# data_loading.py
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from utils import plot_heatmap

class LazyLoadingDataset(Dataset):
    def __init__(self, samples_file, window_size, stride, device):
        self.samples_file = samples_file
        self.window_size = window_size
        self.stride = stride
        self.device = device
        self.samples = None  # Lazy-loaded samples

        # Open the .mat file once to get necessary metadata
        with h5py.File(self.samples_file, "r") as f:
            self.samples = f["samples"][()]
        self.length = self.samples.shape[0] - window_size - 1  # Total number of sequences

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy load data on demand
        i = idx + self.window_size  # Start index for the sample sequence

        # Extract the input/output pairs
        x = self.samples[i - self.window_size : i, :, :]
        y = self.samples[i : i + self.window_size, :, :]

        # Adjust y if it does not have the expected shape
        if y.shape[0] != self.window_size:
            print(f"Warning: y has unexpected shape {y.shape}. Expected shape: ({self.window_size}, 264, 496).")
            pass
            
        # Convert them to tensors and move to the right device
        x_tensor = torch.from_numpy(x).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)

        return x_tensor, y_tensor

def get_data_loaders(samples_file, window_size, stride, batch_size, device):
    dataset = LazyLoadingDataset(samples_file, window_size, stride, device)
    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader


