"""
Transformer Decoder Implementation
--------------------------------

The Transformer decoder is composed of a stack of N identical layers.
Each layer has three sub-layers:
1. Masked multi-head self-attention mechanism
2. Multi-head cross-attention mechanism
3. Position-wise fully connected feed-forward network

Key differences from encoder:
- Uses masked self-attention to prevent attending to future positions
- Has cross-attention layer to attend to encoder outputs
- Each sub-layer employs residual connection and layer normalization

Dimensions Guide:
---------------
- Batch size: B
- Target sequence length: T
- Source sequence length: S
- Model dimension: d_model
- Feed-forward hidden dimension: d_ff
- Number of attention heads: h
- Number of decoder layers: N

Shape transformations:
- Input: (B, T, d_model)
- Self-attention: (B, T, d_model) -> (B, T, d_model)
- Cross-attention: (B, T, d_model) x (B, S, d_model) -> (B, T, d_model)
- Feed-forward: (B, T, d_model) -> (B, T, d_model)
- Output: (B, T, d_model)
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
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # STEP 2: Initialize dropout
        # Create dropout layer with specified rate
        self.dropout = nn.Dropout(dropout)
        
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
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.
    
    Consists of:
    1. Masked self-attention
    2. Cross-attention to encoder output
    3. Feed-forward network
    All wrapped with residual connection and layer normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the decoder layer.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 1: Initialize masked self-attention
        # Create MultiHeadAttention instance for self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # STEP 2: Initialize cross-attention
        # Create MultiHeadAttention instance for cross-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # STEP 3: Initialize feed-forward network
        # Create feed-forward network instance
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # STEP 4: Initialize layer normalizations
        # Three LayerNorm instances: self-attention, cross-attention, feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # STEP 5: Initialize dropout
        # Create dropout layer
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Pass the input through the decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model)
            enc_output (torch.Tensor): Encoder output of shape (B, S, d_model)
            src_mask (torch.Tensor): Source mask of shape (B, T, S)
            tgt_mask (torch.Tensor): Target mask of shape (B, T, T)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model)
        """
        # STEP 6: Apply masked self-attention sub-layer
        # Masked multi-head attention -> Dropout -> Add & Norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # STEP 7: Apply cross-attention sub-layer
        # Cross-attention with encoder output -> Dropout -> Add & Norm
        cross_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # STEP 8: Apply feed-forward sub-layer
        # Feed-forward -> Dropout -> Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Full transformer decoder consisting of N identical layers.
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize the transformer decoder.
        
        Args:
            num_layers (int): Number of decoder layers
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 9: Create decoder layers
        # Create a ModuleList of DecoderLayer instances
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # STEP 10: Initialize dropout
        # Create dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Pass the input through the decoder stack.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model)
            enc_output (torch.Tensor): Encoder output tensor of shape (B, S, d_model)
            src_mask (torch.Tensor): Source mask of shape (B, T, S)
            tgt_mask (torch.Tensor): Target mask of shape (B, T, T)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model)
        """
        # STEP 11: Pass input through each decoder layer in sequence
        # Apply dropout to the input, then pass through layers
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x 