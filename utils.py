# utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity as ssim
from math import log10

def plot_heatmap(data, extent, vmin, vmax, cmap, filename):
    # Create the figure and axis
    fig, ax = plt.subplots()
    extent = [180, 120, -10, 10]
    heatmap = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, extent=extent)
    rect_black = patches.Rectangle(
        (extent[1], extent[2]), extent[0] - extent[1], extent[3] - extent[2],
        linewidth=5, edgecolor='black', facecolor='none')
    ax.add_patch(rect_black)
    rect_white = patches.Rectangle(
        (extent[1], extent[2]), extent[0] - extent[1], extent[3] - extent[2],
        linewidth=4, edgecolor='white', linestyle='--', facecolor='none')
    ax.add_patch(rect_white)
    ax.set_xticks(np.linspace(120, 180, 5))
    ax.set_yticks(np.linspace(-10, 10, 5))
    ax.set_xticklabels([f"{int(label)}°{'W' if label < 0 else 'E'}" for label in np.linspace(120, 180, 5)])
    ax.set_yticklabels([f"{int(label)}°{'S' if label < 0 else 'N'}" for label in np.linspace(-10, 10, 5)])
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.04])
    ax.grid(True, linestyle='--', color='black', alpha=0.5)
    cbar = plt.colorbar(heatmap, cax=cax, orientation='horizontal', pad=-5)
    plt.savefig(filename)
    plt.close()

def add_noise(tensor):
    # Load noise from CSV
    noise_path = 'diffusion_test\\q_sp_final.csv'
    df = pd.read_csv(noise_path, header=None)

    print(tensor.shape, df.shape)

    noise = torch.tensor(df.values, dtype=torch.float32).to(tensor.device)  
    noise = noise.unsqueeze(0).unsqueeze(0).expand(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3])

    return tensor + noise

input_total = 496
app_loss = 0

def app(model, dataloader, device, criterion):
    app_loss = 0

    # Load application data 
    for inputs, targets in dataloader:
        single_input = inputs[0].unsqueeze(0).to(device)  
        y_true = targets[0].unsqueeze(0).to(device) 

    noised_input = add_noise(single_input)
    y_pred = model(single_input)
    y_noise = model(noised_input)

    for i in range(14):
        # Calculate loss
        app_loss = criterion(y_pred[:, i, :, :], y_true[:, i, :, :])
        print(f"Loss at step {i+1}: {app_loss.item()}")

        # Calculate RMSE
        rmse = torch.sqrt(torch.mean((y_pred[:, i, :, :] - y_true[:, i, :, :]) ** 2)).item()
        print(f"RMSE {i+1}: {rmse}")

        # PSNR calculation
        mse = torch.mean((y_pred - y_true) ** 2).item()
        psnr = 20 * log10(1.0 / np.sqrt(mse)) if mse != 0 else 100
        print(f"Origin PSNR {i+1}: {psnr}")
        mse = torch.mean((y_pred - y_noise) ** 2).item()
        psnr = 20 * log10(1.0 / np.sqrt(mse)) if mse != 0 else 100
        print(f"Noise PSNR {i+1}: {psnr}")

        # SSIM calculation
        y_true_np = y_true.squeeze(0).squeeze(0).cpu().detach().numpy()
        y_pred_np = y_pred.squeeze(0).squeeze(0).cpu().detach().numpy()
        ssim_value = ssim(y_true_np, y_pred_np, data_range=y_pred_np.max() - y_pred_np.min())
        print(f"SSIM {i+1}: {ssim_value}")

        # Visualization of predictions
        extent = [180, 120, -10, 10]

        # Predicted heatmap
        plot_heatmap(y_pred[0, i, :, :].cpu().detach().numpy(), extent, vmin=22, vmax=30, cmap='jet', filename=f'diff_img\\predicted_heatmap_{i}.png')

        # True heatmap
        plot_heatmap(y_true[0, i, :, :].cpu().detach().numpy(), extent, vmin=22, vmax=30, cmap='jet', filename=f'diff_img\\true_heatmap_{i}.png')

        # Difference heatmap (Prediction - Ground Truth)
        diff = y_pred[0, i, :, :].cpu().detach().numpy() - y_true[0, i, :, :].cpu().detach().numpy()
        plot_heatmap(diff, extent, vmin=-2, vmax=2, cmap='Spectral', filename=f'diff_img\\heatmap_{i}.png')
