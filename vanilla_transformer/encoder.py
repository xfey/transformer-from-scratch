"""
Transformer Encoder Implementation
--------------------------------

The Transformer encoder is composed of a stack of N identical layers.
Each layer has two sub-layers:
1. Multi-head self-attention mechanism
2. Position-wise fully connected feed-forward network

Each sub-layer employs:
- Residual connection
- Layer normalization

Dimensions Guide:
---------------
- Batch size: B
- Sequence length: L
- Model dimension: d_model
- Feed-forward hidden dimension: d_ff
- Number of attention heads: h
- Number of encoder layers: N

Shape transformations:
- Input: (B, L, d_model)
- Self-attention: (B, L, d_model) -> (B, L, d_model)
- Feed-forward: (B, L, d_model) -> (B, L, d_model)
- Output: (B, L, d_model)
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Transforms the dimension from d_model -> d_ff -> d_model
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model (int): Model's dimension
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 1: Create the two linear transformations
        # First: d_model -> d_ff
        # Second: d_ff -> d_model
        
        
        # STEP 2: Initialize dropout
        # Create dropout layer with specified rate

        
        
    def forward(self, x):
        """
        Pass the input through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 3: Apply the feed-forward transformations
        # First linear -> ReLU -> Dropout -> Second linear
        
        
        return x


class EncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.
    
    Consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    Both wrapped with residual connection and layer normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the encoder layer.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 4: Initialize multi-head self-attention
        # Create MultiHeadAttention instance
        
        
        # STEP 5: Initialize feed-forward network
        # Create FeedForward instance

        
        # STEP 6: Initialize layer normalizations
        # One for attention, one for feed-forward

        
        # STEP 7: Initialize dropout
        # Create dropout layer

        
        
    def forward(self, x, mask=None):
        """
        Pass the input through the encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            mask (torch.Tensor, optional): Mask tensor of shape (B, L, L)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 8: Apply attention sub-layer
        # Multi-head attention -> Dropout -> Add & Norm

        
        # STEP 9: Apply feed-forward sub-layer
        # Feed-forward -> Dropout -> Add & Norm
        
        
        return x


class TransformerEncoder(nn.Module):
    """
    Full transformer encoder consisting of N identical layers.
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the transformer encoder.
        
        Args:
            num_layers (int): Number of encoder layers
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 10: Create encoder layers
        # Create a ModuleList of EncoderLayer instances
        
        
        # STEP 11: Initialize dropout
        # Create dropout layer
        
        
    def forward(self, x, mask=None):
        """
        Pass the input through the encoder stack.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            mask (torch.Tensor, optional): Mask tensor of shape (B, L, L)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 12: Pass input through each encoder layer in sequence
        # Apply dropout to the input, then pass through layers
        
        
        return x
