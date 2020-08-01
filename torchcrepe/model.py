import functools
import os

import numpy as np
import torch
import torch.nn.functional as F

import torchcrepe


###########################################################################
# Model definition
###########################################################################


class Crepe(torch.nn.Module):
    """Crepe model definition"""
    
    def __init__(self, model='full'):
        super().__init__()
        
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f'Model {f} is not supported')
        
        # Overload with eps and momentum conversion given by MMdnn
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d,
                                          eps=0.0010000000474974513,
                                          momentum=0.0)

        # Layer definitions
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=(512, 1),
            stride=(4, 1))
        self.conv1_BN = batch_norm_fn(
            num_features=out_channels[0])
        
        self.conv2 = torch.nn.Conv2d(
            in_channels=in_channels[1],
            out_channels=out_channels[1],
            kernel_size=(64, 1))
        self.conv2_BN = batch_norm_fn(
            num_features=out_channels[1])
        
        self.conv3 = torch.nn.Conv2d(
            in_channels=in_channels[2],
            out_channels=out_channels[2],
            kernel_size=(64, 1))
        self.conv3_BN = batch_norm_fn(
            num_features=out_channels[2])
        
        self.conv4 = torch.nn.Conv2d(
            in_channels=in_channels[3],
            out_channels=out_channels[3],
            kernel_size=(64, 1))
        self.conv4_BN = batch_norm_fn(
            num_features=out_channels[3])
        
        self.conv5 = torch.nn.Conv2d(
            in_channels=in_channels[4],
            out_channels=out_channels[4],
            kernel_size=(64, 1))
        self.conv5_BN = batch_norm_fn(
            num_features=out_channels[4])
        
        self.conv6 = torch.nn.Conv2d(
            in_channels=in_channels[5],
            out_channels=out_channels[5],
            kernel_size=(64, 1))
        self.conv6_BN = batch_norm_fn(
            num_features=out_channels[5])
        
        self.classifier = torch.nn.Linear(
            in_features=self.in_features,
            out_features=torchcrepe.PITCH_BINS)
        
    def forward(self, x, embed=False):
        # Forward pass through first four layers and part of layer five
        x = self.embed(x)
        
        if embed:
            return x
        
        # Forward pass through layer six
        x = self.layer(x, self.conv6, self.conv6_BN)
        
        # shape=(batch, -1)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)
        
        # Compute logits
        return torch.sigmoid(self.classifier(x))
    
    ###########################################################################
    # Forward pass utilities
    ###########################################################################
        
    def embed(self, x):
        """Map input audio to pitch embedding"""
        # shape=(batch, 1, 1024, 1)
        x = x[:, None, :, None]
        
        # Forward pass through first four layers
        x = self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN)
        x = self.layer(x, self.conv3, self.conv3_BN)
        x = self.layer(x, self.conv4, self.conv4_BN)
        
        # Partial forward pass through layer five
        return self.layer(x, self.conv5, self.conv5_BN)
    
    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        """Forward pass through one layer"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return F.max_pool2d(x, (2, 1), (2, 1))
