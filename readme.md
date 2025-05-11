# Diffusion Model with CANDY and UNet for Self-Supervised Learning

## Overview

This project implements a diffusion-based deep learning model for Self-Supervised Learning, combining CANDY (Customizable Attention-based Network for Dynamic data) modules with UNet architectures. The model is designed to handle sequential data with spatial dimensions, particularly suited for tasks like weather prediction or similar spatiotemporal forecasting problems.

## Key Features

- **Diffusion Process**: Implements both forward (noise addition) and reverse (denoising) diffusion processes
- **Hybrid Architecture**: Combines CANDY modules for feature processing with UNet for reconstruction
- **Spatiotemporal Handling**: Processes 4D tensors (batch × channels × height × width) with temporal sequences
- **Custom Components**: Includes specialized activation functions and attention mechanisms
- **Evaluation Metrics**: Computes RMSE, PSNR, and SSIM for performance assessment

## Model Architecture

### Main Components

1. **DiffusionModel**:
   - Manages the complete diffusion process with T steps
   - Contains T CANDY modules and T UNet modules (one for each step)

2. **CANDY**:
   - Custom architecture with learnable masks and attention mechanisms
   - Processes input through parallel pathways (p_set and z_output)
   - Uses a custom activation function

3. **UNet**:
   - Modified version of the standard UNet architecture
   - Implements skip connections and up/down sampling
   - Configurable for bilinear or transposed convolution upsampling

## Data Handling

- **LazyLoadingDataset**: Efficiently loads and processes large spatiotemporal datasets
- **Windowed Sequences**: Creates input-output pairs with configurable window sizes
- **Train/Val/Test Split**: Automatic dataset splitting with 90%/5%/5% ratio

## Training Process

- **Custom Loss Functions**: Primarily uses MSE loss
- **Checkpointing**: Saves and loads model states for training continuation
- **Early Stopping**: Monitors loss for potential early termination
- **Device-Agnostic**: Runs on both CPU and GPU (CUDA)

## Evaluation Metrics

The model computes several performance metrics:
- Root Mean Square Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Visualization of prediction differences via heatmaps

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- h5py (for .mat file handling)
- pandas
- scikit-image (for SSIM calculation)

## Usage

1. **Data Preparation**:
   - Prepare your data in .mat format with 'samples' as the key
   - Update the `samples_file` path in `main.py`

2. **Training**:
   ```bash
   python main.py

3. **Configuration**:
   - Adjust hyperparameters in main.py (batch size, learning rate, etc.)
   - Modify model architecture in respective files

4. **Visualization**:
   - Heatmaps and comparisons are automatically saved during evaluation

## File Structure
`main.py`: Main training script
`data_loading.py`: Dataset handling and loader creation
`train.py`: Training and validation logic
`utils.py`: Visualization and evaluation functions
`models`/:
   `diffusion.py`: Main diffusion model
   `candy.py`: CANDY module implementation
   `unet.py`: UNet implementation

## Notes
The current implementation uses a batch size of 1 due to memory constraints with large spatiotemporal data
Model parameters should be adjusted based on your specific dataset characteristics
The UNet implementation is currently using a simplified version (commented code shows full version)

## Future Work
Implement more sophisticated diffusion scheduling
Add support for larger batch sizes
Experiment with different attention mechanisms
Optimize memory usage for larger datasets