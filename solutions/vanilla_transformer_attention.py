"""
Attention Mechanism Implementation
--------------------------------

The attention mechanism is a core component of the Transformer architecture, introduced in 
"Attention Is All You Need" (Vaswani et al., 2017).

Key Concepts:
------------
1. Scaled Dot-Product Attention:
   - Input: queries (Q), keys (K), values (V)
   - Output: weighted sum of values
   - Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
   
2. Multi-Head Attention:
   - Allows model to attend to information from different positions
   - Each head learns different representation subspaces
   - Concatenates and projects the outputs of multiple attention heads

Dimensions Guide:
---------------
For a single attention head:
- Input sequence length: L
- Batch size: B
- Model dimension: d_model
- Key dimension: d_k = d_model/num_heads
- Value dimension: d_v = d_model/num_heads

Shape transformations:
- Q, K, V initial: (B, L, d_model)
- After projection: (B, L, d_k) for Q,K; (B, L, d_v) for V
- Attention scores: (B, L, L)
- Output: (B, L, d_v)

For multi-head attention:
- Each head processes: (B, L, d_k)
- After concatenation: (B, L, num_heads * d_v)
- Final output: (B, L, d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.
    
    This is the core attention mechanism that computes:
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, d_k):
        """
        Initialize the attention module.
        
        Args:
            d_k (int): Dimension of keys/queries
        """
        super().__init__()
        # Store the key dimension for scaling
        # STEP 1: Save d_k and calculate the scaling factor √d_k
        self.d_k = d_k
        self.scale = math.sqrt(d_k)
        
    def forward(self, Q, K, V, mask=None):
        """
        Compute the attention weights and output.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_k)
            K (torch.Tensor): Keys tensor of shape (B, L, d_k)
            V (torch.Tensor): Values tensor of shape (B, L, d_v)
            mask (torch.Tensor, optional): Mask tensor of shape (B, L, L)
        
        Returns:
            tuple: (output, attention_weights)
                - output shape: (B, L, d_v)
                - attention_weights shape: (B, L, L)
        """
        # STEP 2: Get batch size and sequence length
        # Hint: Use Q.size()
        batch_size, seq_length, _ = Q.size()
        
        # STEP 3: Compute attention scores
        # Formula: scores = Q × K^T
        # Expected shape: (B, L, L)
        scores = torch.bmm(Q, K.transpose(1, 2))
        
        # STEP 4: Scale the scores
        # Formula: scores = scores / √d_k
        scores = scores / self.scale
        
        # STEP 5: Apply mask if provided
        # Set masked positions to -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # STEP 6: Apply softmax to get attention weights
        # Use F.softmax()
        attention_weights = F.softmax(scores, dim=-1)
        
        # STEP 7: Compute output
        # Formula: output = attention_weights × V
        # Expected shape: (B, L, d_v)
        output = torch.bmm(attention_weights, V)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    This module splits the queries, keys and values into multiple heads,
    applies scaled dot-product attention independently on each head,
    and concatenates the results.
    """
    
    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention module.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # STEP 1: Save parameters
        # Store d_model, num_heads, and compute d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # STEP 2: Initialize projection matrices
        # Create W_q, W_k, W_v for all heads
        # Shape: (d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # STEP 3: Initialize output projection
        # Create W_o to project concatenated heads
        # Shape: (d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # STEP 4: Initialize attention module
        # Create ScaledDotProductAttention instance
        self.attention = ScaledDotProductAttention(self.d_k)
        
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        and transpose the result to (B, num_heads, L, d_k)
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
        
        Returns:
            torch.Tensor: Reshaped tensor of shape (B, num_heads, L, d_k)
        """
        # STEP 5: Implement head splitting
        # Reshape x from (B, L, d_model) to (B, L, num_heads, d_k), 
        # then transpose dimensions to (B, num_heads, L, d_k)
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(1, 2)
        
    def combine_heads(self, x):
        """
        Transpose and reshape the input from (B, num_heads, L, d_k)
        back to (B, L, d_model)
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, num_heads, L, d_k)
        
        Returns:
            torch.Tensor: Reshaped tensor of shape (B, L, d_model)
        """
        # STEP 6: Implement head combining
        # Transpose and reshape dimensions
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """
        Compute multi-head attention.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_model)
            K (torch.Tensor): Keys tensor of shape (B, L, d_model)
            V (torch.Tensor): Values tensor of shape (B, L, d_model)
            mask (torch.Tensor, optional): Mask tensor of shape (B, L, L)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 7: Apply query, key, value projections
        # Use the projection matrices initialized in __init__
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # STEP 8: Split heads
        # Use the split_heads method
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # STEP 9: Apply scaled dot-product attention
        # Use the attention module initialized in __init__
        output = self.attention(Q, K, V, mask)
        
        # STEP 10: Combine heads
        # Use the combine_heads method
        output = self.combine_heads(output)
        
        # STEP 11: Apply output projection
        # Project back to d_model dimensions
        output = self.W_o(output)
        
        return output
