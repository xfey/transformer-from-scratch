"""
Linear Attention Implementation
-----------------------------

Linear Attention is a variant of the attention mechanism that reduces the computational complexity
from O(L²) to O(L), where L is the sequence length. This makes it particularly suitable for
processing long sequences.

Paper: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (Katharopoulos et al., 2020)

Key Concepts:
------------
1. Kernel Feature Maps:
   - Replaces the softmax with a kernel function: K(x, y) = φ(x)ᵀφ(y)
   - Uses the associative property of matrix multiplication to change the order of operations
   - Reduces complexity from O(L²) to O(L)

2. Linear Attention Formula:
   - Standard attention: Attention(Q,K,V) = softmax(QKᵀ)V
   - Linear attention: Attention(Q,K,V) = φ(Q)(φ(K)ᵀV) / (φ(Q)φ(K)ᵀ1)
   - Where φ is a feature map (e.g., elu(x) + 1)

3. Causal Masking:
   - For autoregressive models, causal masking can be implemented efficiently
   - Maintains the linear complexity advantage

Dimensions Guide:
---------------
- Batch size: B
- Sequence length: L
- Model dimension: d_model
- Key/Query dimension: d_k = d_model/num_heads
- Value dimension: d_v = d_model/num_heads

Shape transformations:
- Q, K, V initial: (B, L, d_k) or (B, L, d_v)
- After feature map: (B, L, d_k) or (B, L, d_v)
- KᵀV computation: (B, d_k, d_v)
- Final output: (B, L, d_v)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def elu_feature_map(x):
    """
    Feature map for the linear attention using ELU activation.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Transformed tensor with feature map applied
    """
    # STEP 1: Apply the ELU+1 feature map
    # Formula: φ(x) = elu(x) + 1
    x = F.elu(x) + 1
    
    return x


class LinearAttention(nn.Module):
    """
    Linear Attention module with O(L) complexity instead of O(L²).
    
    This module implements attention using the kernel trick to avoid
    computing the full attention matrix.
    """
    
    def __init__(self, d_model, d_k=None, feature_map=None, causal=False, eps=1e-6):
        """
        Initialize the linear attention module.
        
        Args:
            d_model (int): Model dimension
            d_k (int, optional): Key dimension. If None, set to d_model
            feature_map (callable, optional): Feature map function. If None, use elu_feature_map
            causal (bool): Whether to use causal masking
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        # STEP 2: Save parameters
        # Store dimensions, feature map function, causal flag, and epsilon
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.feature_map = feature_map if feature_map is not None else elu_feature_map
        self.causal = causal
        self.eps = eps
        
    def forward(self, Q, K, V):
        """
        Compute linear attention.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_k)
            K (torch.Tensor): Keys tensor of shape (B, L, d_k)
            V (torch.Tensor): Values tensor of shape (B, L, d_v)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_v)
        """
        # STEP 3: Get dimensions
        # Extract B, L, d_k, d_v from input tensors
        B, L, d_k = Q.shape
        _, _, d_v = V.shape
        
        # STEP 4: Apply feature map to queries and keys
        # Transform Q and K using the feature map
        Q = self.feature_map(Q)  # (B, L, d_k)
        K = self.feature_map(K)  # (B, L, d_k)
        
        # STEP 5: Compute attention for non-causal case
        # If not causal, compute KᵀV first, then multiply by Q
        if not self.causal:
            # STEP 5.1: Compute KᵀV
            # Formula: KV = φ(K)ᵀV, shape: # (B, d_k, d_v)
            KV = torch.bmm(K.transpose(1, 2), V)  # (B, d_k, d_v)
            
            # STEP 5.2: Compute normalization factor
            # Formula: Z = φ(K)ᵀ1 (sum over keys)
            Z = K.sum(dim=1, keepdim=True)  # (B, 1, d_k)
            
            # STEP 5.3: Compute output
            # Formula: φ(Q)KV / (φ(Q)Z + ε)
            output = torch.bmm(Q, KV)  # (B, L, d_v)
            normalizer = torch.bmm(Q, Z.transpose(1, 2)) + self.eps  # (B, L, 1)
            output = output / normalizer
            
        # STEP 6: Compute attention for causal case
        # If causal, use cumulative sum for efficient causal masking
        else:
            # STEP 6.1: Initialize output and normalization tensors with zero
            output = torch.zeros(B, L, d_v, device=Q.device)
            Z = torch.zeros(B, L, 1, device=Q.device)
            
            # STEP 6.2: Iterate through sequence positions
            # For each position, compute attention only with previous positions
            KV_cumsum = torch.zeros(B, d_k, d_v, device=Q.device)
            K_cumsum = torch.zeros(B, d_k, 1, device=Q.device)
            
            # causal attention
            for l in range(L):
                KV_cumsum = KV_cumsum + torch.bmm(
                    K[:, l:l+1].transpose(1, 2), V[:, l:l+1]
                )
                output[:, l] = torch.bmm(Q[:, l:l+1], KV_cumsum).squeeze(1)
                
                K_cumsum = K_cumsum + K[:, l:l+1].transpose(1, 2).sum(dim=2, keepdim=True)
                Z[:, l] = torch.bmm(Q[:, l:l+1], K_cumsum).squeeze(1)
            
            output = output / (Z + self.eps)
            
        return output


class LinearMultiHeadAttention(nn.Module):
    """
    Multi-Head Linear Attention module.
    
    This module splits the queries, keys and values into multiple heads,
    applies linear attention independently on each head,
    and concatenates the results.
    """
    
    def __init__(self, d_model, num_heads, causal=False):
        """
        Initialize the multi-head linear attention module.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            causal (bool): Whether to use causal masking
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # STEP 7: Save parameters
        # Store d_model, num_heads, and compute d_k
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # STEP 8: Initialize projection matrices
        # Create W_q, W_k, W_v for all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # STEP 9: Initialize output projection
        # Create W_o to project concatenated heads
        self.W_o = nn.Linear(d_model, d_model)
        
        # STEP 10: Initialize attention modules
        # Create LinearAttention instances for each head
        # Hint: For single head: using d_model. For multihead (here): using d_k instead
        self.attention_heads = nn.ModuleList([
            LinearAttention(self.d_k, causal=causal)
            for _ in range(num_heads)
        ])
        
    def forward(self, Q, K, V):
        """
        Compute multi-head linear attention.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_model)
            K (torch.Tensor): Keys tensor of shape (B, L, d_model)
            V (torch.Tensor): Values tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 11: Get batch size and sequence length
        # Extract B and L from input tensors
        B, L, _ = Q.shape
        
        # STEP 12: Apply query, key, value projections
        # Project and reshape to (B, num_heads, L, d_k)
        # Hint: reshape from (B, L, d_model) to (B, L, num_heads, d_k) fisrt, then transpose
        Q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        K = self.W_k(K).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        V = self.W_v(V).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        
        # STEP 13: Apply linear attention for each head
        # Process each head with its own LinearAttention instance
        head_outputs = []
        for h in range(self.num_heads):
            head_outputs.append(
                self.attention_heads[h](
                    Q[:, h],  # (B, L, d_k)
                    K[:, h],  # (B, L, d_k)
                    V[:, h]   # (B, L, d_k)
                )
            )
        
        # STEP 14: Concatenate and project outputs
        # Combine head outputs to (B, L, d_model), then apply output projection
        multi_head_output = torch.cat(head_outputs, dim=-1)  # (B, L, d_model)
        output = self.W_o(multi_head_output)  # (B, L, d_model)
        
        return output
