"""
Swin Transformer Implementation
-----------------------------

Swin Transformer is a hierarchical vision transformer that uses shifted windows for self-attention.
Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)

https://www.zhihu.com/tardis/zm/art/577855860?source_id=1003

Key Concepts:
------------
1. Hierarchical Feature Maps:
   - Starts with small-sized patches (e.g., 4×4)
   - Progressively merges neighboring patches
   - Creates a hierarchical representation

2. Window-based Self-Attention:
   - Computes self-attention within local windows
   - Reduces computational complexity from quadratic to linear
   - Uses shifted windows between layers for cross-window connections

3. Relative Position Bias:
   - Adds learnable relative position embeddings
   - Considers both row and column-wise relative positions
   - Shared across heads in the same layer

Dimensions Guide:
---------------
- Batch size: B
- Input image size: (H, W)
- Patch size: P (e.g., 4×4)
- Window size: M×M
- Number of heads: num_heads
- Hidden dimension: dim
- MLP ratio: mlp_ratio (typically 4)

Shape transformations:
- Input image: (B, C, H, W)
- After patching: (B, H/P, W/P, C*P*P)
- Within window: (B*num_windows, M*M, dim)
- After window attention: (B, H/P, W/P, dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Window size M
        
    Returns:
        windows: (num_windows*B, M, M, C)
    """
    # STEP 1: Get input dimensions
    # Extract B, H, W, C from input tensor
    B, H, W, C = x.shape
    
    # STEP 2: Reshape input
    # Reshape to (B, H/M, M, W/M, M, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    
    # STEP 3: Permute and reshape to windows
    # Permute: (B, H/M, W/M, M, M, C)
    # Final shape: (B*num_windows, M, M, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    
    Args:
        windows (torch.Tensor): Input windows of shape (num_windows*B, M, M, C)
        window_size (int): Window size M
        H (int): Height of feature map
        W (int): Width of feature map
        
    Returns:
        x: (B, H, W, C)
    """
    # STEP 1: Get dimensions
    # Calculate B from input dimensions
    B = int(windows.shape[0] // (H * W // window_size // window_size))
    
    # STEP 2: Reshape windows
    # Reshape to (B, H/M, W/M, M, M, C)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    
    # STEP 3: Permute and reshape to image
    # Final shape: (B, H, W, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-attention module.
    Includes relative position bias.
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        """
        Initialize the window attention module.
        
        Args:
            dim (int): Input dimension
            window_size (tuple[int]): Window size (M, M)
            num_heads (int): Number of attention heads
            qkv_bias (bool): Add bias to qkv projection
        """
        super().__init__()
        # STEP 1: Save parameters
        # Store window_size, dim, and num_heads
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # STEP 2: Create qkv projection
        # Single matrix for query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # STEP 3: Create attention projection
        # Project attention output back to dim
        self.proj = nn.Linear(dim, dim)
        
        # STEP 4: Create relative position bias
        # Initialize relative position bias table and index
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        # Get coords for all pixels
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        
        # Calculate the flatten coords
        coords_flatten = torch.flatten(coords, 1)
        
        # Calculate relative coords
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        
        # Register buffer for index
        self.register_buffer("relative_position_index", relative_position_index)
        # Init bias_table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B*num_windows, M*M, C)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor of shape (B*num_windows, M*M, C)
        """
        # STEP 5: Get input dimensions
        # Extract B_, N, C from input tensor
        B_, N, C = x.shape
        
        # STEP 6: Generate QKV matrices
        # Project x to query, key, value
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # STEP 7: Compute attention with relative position bias
        # Calculate attention scores and apply mask if provided
        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # if mask is not None, apply mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # STEP 8: Apply attention to values
        # Combine attention weights with values and project
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with window attention and shifted window attention.
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        """
        Initialize Swin Transformer block.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            window_size (int): Window size
            shift_size (int): Shift size for SW-MSA
            mlp_ratio (float): MLP expansion ratio
            qkv_bias (bool): Add bias to qkv projection
            drop (float): Dropout rate
            attn_drop (float): Attention dropout rate
        """
        super().__init__()
        # STEP 1: Save parameters
        # Store dimensions and sizes
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # STEP 2: Create layer normalization
        # Two LayerNorm layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # STEP 3: Create window attention
        # Initialize WindowAttention module
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size),
            num_heads=num_heads, qkv_bias=qkv_bias)
        
        # STEP 4: Create MLP
        # Two-layer MLP with GELU activation
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
        )
        
        # STEP 5: Initialize dropout
        # Create dropout layers
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, H, W):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, H*W, C)
            H (int): Height of feature map
            W (int): Width of feature map
            
        Returns:
            torch.Tensor: Output tensor of shape (B, H*W, C)
        """
        # STEP 6: Apply first LayerNorm and reshape
        # Prepare input for attention
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # STEP 7: Cyclic shift (if shift_size > 0)
        # Shift windows if needed
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # STEP 8: Window partition
        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # STEP 9: Window attention
        # Apply attention within windows
        attn_windows = self.attn(x_windows)
        
        # STEP 10: Reverse window partition
        # Merge windows back to feature map
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # STEP 11: Reverse cyclic shift
        # Reverse the shift if applied
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # STEP 12: Apply MLP
        # Second LayerNorm and MLP
        x = x.view(B, H * W, C)
        x = shortcut + self.drop(x)
        x = x + self.drop(self.mlp(self.norm2(x)))
        
        return x


class SwinTransformer(nn.Module):
    """
    Full Swin Transformer model for image classification.
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.):
        """
        Initialize Swin Transformer.
        
        Args:
            img_size (int): Input image size
            patch_size (int): Patch size
            in_chans (int): Number of input channels
            num_classes (int): Number of classes
            embed_dim (int): Initial embedding dimension
            depths (list[int]): Number of blocks in each stage
            num_heads (list[int]): Number of attention heads in each stage
            window_size (int): Window size
            mlp_ratio (float): MLP expansion ratio
            qkv_bias (bool): Add bias to qkv projection
            drop_rate (float): Dropout rate
            attn_drop_rate (float): Attention dropout rate
        """
        super().__init__()
        # STEP 1: Save parameters
        # Store model configuration
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # STEP 2: Create patch embedding
        # Convert image to patch embeddings
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
        # STEP 3: Create Swin Transformer blocks
        # Create multiple stages of Swin blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate)
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            if i_layer < self.num_layers - 1:
                self.layers.append(
                    nn.Conv2d(int(embed_dim * 2 ** i_layer),
                             int(embed_dim * 2 ** (i_layer + 1)),
                             kernel_size=2, stride=2)
                )
        
        # STEP 4: Create classification head
        # Final layer normalization and classifier
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        # STEP 5: Patch embedding
        # Convert image to patch embeddings
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        # STEP 6: Pass through Swin blocks
        # Process through each stage
        for i in range(self.num_layers):
            layer = self.layers[i * 2]
            for block in layer:
                x = block(x, H, W)
                
            if i < self.num_layers - 1:
                x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
                x = self.layers[i * 2 + 1](x)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
        
        # STEP 7: Global average pooling
        # Pool features
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        
        # STEP 8: Classification
        # Get final predictions
        x = self.head(x)
        
        return x 