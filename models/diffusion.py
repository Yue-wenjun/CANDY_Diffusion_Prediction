# models.py
import torch
import torch.nn as nn
from models.candy import CANDY
from models.unet import UNet

import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size, T):
        super(DiffusionModel, self).__init__()
        self.in_channel = in_channel
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.T = T  # Number of diffusion steps

        # Create a list of CANDY modules, one for each time step
        self.candies = nn.ModuleList([
            CANDY(batch_size, in_channel, hidden_channel, out_channel, input_size, hidden_size)
            for _ in range(T)
        ])
        
        # Create a list of UNet modules, one for each time step
        self.unets = nn.ModuleList([
            UNet(in_channel, out_channel)
            for _ in range(T)
        ])

    def forward(self, x, graph_schedule=None):
        device = x.device
        origin = torch.zeros(self.T, self.in_channel, self.hidden_size, self.input_size).to(device)
        input = x

        # Forward diffusion process (adding noise in each step)
        for t in range(self.T):
            # Pass through the t-th CANDY module (feature processing)
            output, origin_t = self.candies[t](input)
            origin[t] = origin_t
            input = output

        if graph_schedule is None:
            graph_schedule = torch.linspace(0.5, 0.5, self.T).to(device)

        # Reverse diffusion process (denoising)
        for t in reversed(range(self.T)):
            graph_factor = graph_schedule[t]
            reverse_input = (1 - graph_factor) * input + graph_factor * origin[t]

            # Pass through the t-th UNet module for reconstruction
            output = self.unets[t](reverse_input)
            input = output  # Update input for the next step

        output = 0.6 * output + 0.4 * x

        return output