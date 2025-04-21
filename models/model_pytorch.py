"""PyTorch model for point cloud autoencoder. PointNet encoder, FC decoder.
Using Chamfer distance loss.

Converted from the TensorFlow implementation by Charles R. Qi
Author: Arvind Car
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Simple 2D convolution module with batch normalization
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', bn=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0 if padding=='valid' else padding)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        
    def forward(self, x):
        x = self.conv(x).contiguous()
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x)

# PointNet encoder
class PointNetEncoder(nn.Module):
    def __init__(self, point_dim=3):
        super(PointNetEncoder, self).__init__()
        
        # Encoder layers
        self.conv1 = Conv2d(1, 64, [1, point_dim])
        self.conv2 = Conv2d(64, 64, [1, 1])
        self.conv3 = Conv2d(64, 64, [1, 1])
        self.conv4 = Conv2d(64, 128, [1, 1])
        self.conv5 = Conv2d(128, 1024, [1, 1])
        
    def forward(self, point_cloud):
        """
        Input:
            point_cloud: tensor (B, N, 3)
        Output:
            global_feat: tensor (B, 1024), global feature vector
            end_points: dict containing intermediate features
        """
        batch_size = point_cloud.shape[0]
        num_point = point_cloud.shape[1]
        end_points = {}
        
        # Add channel dimension for 2D convolutions
        input_image = point_cloud.unsqueeze(-1).permute(0, 3, 1, 2)  # B,1,N,3
        
        # Encoder
        net = self.conv1(input_image)  # B,64,N,1
        net = self.conv2(net)  # B,64,N,1
        point_feat = self.conv3(net)  # B,64,N,1
        end_points['point_features'] = point_feat
        
        net = self.conv4(point_feat)  # B,128,N,1
        net = self.conv5(net)  # B,1024,N,1
        
        # Global feature - max pooling
        global_feat = torch.max(net, dim=2, keepdim=True)[0]  # B,1024,1,1
        global_feat = global_feat.view(batch_size, -1)  # B,1024
        
        end_points['embedding'] = global_feat
        return global_feat, end_points

# FC decoder for point clouds
class PointNetFCDecoder(nn.Module):
    def __init__(self, num_point=1024, latent_dim=1024):
        super(PointNetFCDecoder, self).__init__()
        self.num_point = num_point
        
        # Decoder layers
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(1024, num_point * 3)
        
    def forward(self, global_feat):
        """
        Input:
            global_feat: tensor (B, latent_dim), global feature vector
        Output:
            reconstructed_points: tensor (B, N, 3), reconstructed point cloud
        """
        batch_size = global_feat.shape[0]
        
        # Decoder (FC)
        net = self.fc1(global_feat)
        net = self.fc2(net)
        net = self.fc3(net)  # B,N*3
        
        reconstructed_points = net.view(batch_size, self.num_point, 3)
        return reconstructed_points

# Combined autoencoder model
class PointNetAE(nn.Module):
    def __init__(self, num_point=1024, point_dim=3, latent_dim=1024):
        super(PointNetAE, self).__init__()
        self.encoder = PointNetEncoder(point_dim)
        self.decoder = PointNetFCDecoder(num_point, latent_dim)
        
    def forward(self, point_cloud):
        """
        Input:
            point_cloud: tensor (B, N, 3)
        Output:
            reconstructed_points: tensor (B, N, 3)
            end_points: dict containing intermediate features
        """
        # Encode
        global_feat, end_points = self.encoder(point_cloud)
        
        # Decode
        reconstructed_points = self.decoder(global_feat)
        
        return reconstructed_points, end_points

def chamfer_distance(point_cloud1, point_cloud2):
    """
    Calculate Chamfer Distance between two point clouds
    Input:
        point_cloud1: tensor (B, N, 3)
        point_cloud2: tensor (B, M, 3)
    Output:
        chamfer_dist: tensor (B,)
    """
    # For each point in point_cloud1, find its nearest neighbor in point_cloud2
    point_cloud1 = point_cloud1.unsqueeze(2)  # B,N,1,3
    point_cloud2 = point_cloud2.unsqueeze(1)  # B,1,M,3
    
    # Calculate pairwise distances
    dist = torch.sum((point_cloud1 - point_cloud2) ** 2, dim=-1)  # B,N,M
    
    # Get minimum distances in both directions
    dist1 = torch.min(dist, dim=2)[0]  # B,N
    dist2 = torch.min(dist, dim=1)[0]  # B,M
    
    # Sum over points
    chamfer_dist = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)  # B
    
    return chamfer_dist

def get_loss(pred, label, end_points=None):
    """
    Input:
        pred: tensor (B, N, 3) - predicted point cloud
        label: tensor (B, N, 3) - ground truth point cloud
        end_points: dict
    Output:
        loss: tensor - chamfer distance loss
    """
    if end_points is None:
        end_points = {}
    
    chamfer_loss = chamfer_distance(pred, label)
    loss = torch.mean(chamfer_loss)
    
    # Scale by 100 to match original implementation
    end_points['pcloss'] = loss
    return loss * 100, end_points


if __name__ == '__main__':
    # Test the implementation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = torch.zeros((32, 1024, 3)).to(device)
    
    # Test individual components
    encoder = PointNetEncoder().to(device)
    decoder = PointNetFCDecoder(num_point=1024).to(device)
    
    # Test encoder
    global_feat, encoder_endpoints = encoder(inputs)
    print(f"Input shape: {inputs.shape}")
    print(f"Global feature shape: {global_feat.shape}")
    
    # Test decoder
    reconstructed = decoder(global_feat)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test full model
    model = PointNetAE(num_point=1024).to(device)
    outputs, end_points = model(inputs)
    print(f"Full model output shape: {outputs.shape}")
    print(f"Embedding shape: {end_points['embedding'].shape}")
    
    # Test loss
    loss, _ = get_loss(outputs, torch.zeros((32, 1024, 3)).to(device), end_points)
    print(f"Loss: {loss.item()}")