"""
Performer Implementation
-----------------------------

Performer is an efficient Transformer variant that uses Fast Attention Via Positive Orthogonal Random Features (FAVOR+)
to approximate the attention mechanism with linear complexity O(L) instead of quadratic O(L²).

Paper: "Rethinking Attention with Performers" (Choromanski et al., 2020)

Key Concepts:
------------
1. FAVOR+ (Fast Attention Via Positive Orthogonal Random Features):
   - Approximates softmax attention using random feature maps
   - Leverages the associative property of matrix multiplication
   - Reduces complexity from O(L²) to O(L)

2. Random Feature Maps:
   - Uses random projections to approximate the softmax kernel
   - Positive orthogonal random features improve numerical stability
   - Unbiased estimation of the original attention

3. Kernelized Attention:
   - Views attention as a kernel function: K(Q, K) = exp(QK^T / sqrt(d))
   - Approximates this kernel with random features: φ(Q)φ(K)^T ≈ K(Q, K)
   - Allows rewriting attention as: Attention(Q,K,V) ≈ φ(Q)(φ(K)^TV) / (φ(Q)φ(K)^T1)

Dimensions Guide:
---------------
- Batch size: B
- Sequence length: L
- Model dimension: d_model
- Key/Query dimension: d_k = d_model/num_heads
- Value dimension: d_v = d_model/num_heads
- Number of random features: m (typically 256-512)

Shape transformations:
- Q, K, V initial: (B, L, d_k) or (B, L, d_v)
- After feature map: (B, L, m)
- KᵀV computation: (B, m, d_v)
- Final output: (B, L, d_v)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def orthogonal_random_matrix(nrows, ncols, device=None):
    """
    Create a random orthogonal matrix for FAVOR+.
    
    Args:
        nrows (int): Number of rows
        ncols (int): Number of columns
        device (torch.device): Device to create the matrix on
        
    Returns:
        torch.Tensor: Random orthogonal matrix
    """
    # STEP 1: Create a random matrix
    # Generate a random Gaussian matrix
    q = torch.randn(nrows, ncols, device=device)
    
    # STEP 2: Apply QR decomposition
    # Get the orthogonal component
    q, _ = torch.linalg.qr(q)
    
    # STEP 3: Ensure proper shape
    # Adjust dimensions if needed
    if nrows < ncols:
        q = q.T
    
    return q


class PerformerAttention(nn.Module):
    """
    Performer attention module using FAVOR+ for efficient attention computation.
    
    This module approximates the softmax attention using random feature maps
    to achieve linear complexity in sequence length.
    """
    
    def __init__(self, d_model, num_features=256, ortho_features=True, redraw=False, eps=1e-6):
        """
        Initialize the Performer attention module.
        
        Args:
            d_model (int): Model dimension
            num_features (int): Number of random features for approximation
            ortho_features (bool): Whether to use orthogonal random features
            redraw (bool): Whether to redraw random features for each forward pass
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        # STEP 4: Save parameters
        self.d_model = d_model
        self.num_features = num_features
        self.ortho_features = ortho_features
        self.redraw = redraw
        self.eps = eps
        
        # STEP 5: Initialize projection matrices
        # Create random projections for feature maps
        self.create_projection = orthogonal_random_matrix if ortho_features \
            else lambda n, d, device: torch.randn(n, d, device=device)
        
        if not redraw:
            self.register_buffer("projection_matrix", self.create_projection(num_features, d_model, None))
        
    def _get_feature_map(self, x, is_query):
        """
        Apply random feature map to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            is_query (bool): Whether the input is a query
            
        Returns:
            torch.Tensor: Transformed tensor with feature map applied
        """
        # STEP 6: Compute random features
        # Project input to random feature space
        projection_matrix = self.projection_matrix if not self.redraw else self.create_projection(
            self.num_features, self.d_model, x.device)
        
        # Normalize x by sqrt(d) for stability
        x = x / math.sqrt(self.d_model)
        
        # Project to random feature space
        # random_features = torch.einsum('...d,md->...m', x, projection_matrix)
        random_features = torch.matmul(x, projection_matrix.transpose(0, 1))
        
        # STEP 7: Apply nonlinearity
        # Transform features
        # For queries: exp(x) / sqrt(m), For keys: exp(x)
        if is_query:
            x_feature = torch.exp(random_features) / math.sqrt(self.num_features)
        else:
            x_feature = torch.exp(random_features)
        
        return x_feature
    
    def forward(self, Q, K, V):
        """
        Compute approximate attention using FAVOR+.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_model)
            K (torch.Tensor): Keys tensor of shape (B, L, d_model)
            V (torch.Tensor): Values tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 8: Apply feature maps
        # Transform Q and K using random feature maps
        Q_feature = self._get_feature_map(Q, is_query=True)  # (B, L_Q, m)
        K_feature = self._get_feature_map(K, is_query=False)  # (B, L_K, m)
        
        # STEP 9: Compute attention
        # Formula: φ(Q)(φ(K)ᵀV) / (φ(Q)φ(K)ᵀ1)
        
        # Compute K_feature^T * V
        KV = torch.bmm(K_feature.transpose(1, 2), V)  # (B, m, d_v)
        
        # Compute normalization factor: K_feature^T * 1
        Z = K_feature.sum(dim=1, keepdim=True)  # (B, 1, m)
        
        # Compute output: Q_feature * KV / (Q_feature * Z)
        output = torch.bmm(Q_feature, KV)  # (B, L_Q, d_v)
        normalizer = torch.bmm(Q_feature, Z.transpose(1, 2)) + self.eps  # (B, L_Q, 1)
        output = output / normalizer
        
        return output


class PerformerMultiHeadAttention(nn.Module):
    """
    Multi-Head Performer Attention module.
    
    This module splits the queries, keys and values into multiple heads,
    applies Performer attention independently on each head,
    and concatenates the results.
    """
    
    def __init__(self, d_model, num_heads, num_features=256, ortho_features=True, redraw=False):
        """
        Initialize the multi-head Performer attention module.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            num_features (int): Number of random features for approximation
            ortho_features (bool): Whether to use orthogonal random features
            redraw (bool): Whether to redraw random features for each forward pass
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # STEP 10: Save parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # STEP 11: Initialize projection matrices
        # Create W_q, W_k, W_v for all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # STEP 12: Initialize output projection
        # Create W_o to project concatenated heads
        self.W_o = nn.Linear(d_model, d_model)
        
        # STEP 13: Initialize attention modules
        # Create PerformerAttention instances for each head
        self.attention_heads = nn.ModuleList([
            PerformerAttention(
                self.d_k, 
                num_features=num_features, 
                ortho_features=ortho_features, 
                redraw=redraw
            )
            for _ in range(num_heads)
        ])
        
    def forward(self, Q, K, V):
        """
        Compute multi-head Performer attention.
        
        Args:
            Q (torch.Tensor): Queries tensor of shape (B, L, d_model)
            K (torch.Tensor): Keys tensor of shape (B, L, d_model)
            V (torch.Tensor): Values tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 14: Get batch size and sequence length
        # Extract B and L from input tensors
        B, L, _ = Q.shape
        
        # STEP 15: Apply query, key, value projections
        # Project and reshape to (B, num_heads, L, d_k)
        Q = self.W_q(Q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        K = self.W_k(K).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        V = self.W_v(V).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, L, d_k)
        
        # STEP 16: Apply Performer attention for each head
        # Process each head with its own PerformerAttention instance
        head_outputs = []
        for h in range(self.num_heads):
            head_outputs.append(
                self.attention_heads[h](
                    Q[:, h],  # (B, L, d_k)
                    K[:, h],  # (B, L, d_k)
                    V[:, h]   # (B, L, d_k)
                )
            )
        
        # STEP 17: Concatenate and project outputs
        # Combine head outputs and apply output projection
        multi_head_output = torch.cat(head_outputs, dim=-1)  # (B, L, d_model)
        output = self.W_o(multi_head_output)  # (B, L, d_model)
        
        return output


class PerformerEncoderLayer(nn.Module):
    """
    Single layer of the Performer encoder.
    
    Consists of:
    1. Multi-head Performer attention
    2. Feed-forward network
    Both wrapped with residual connection and layer normalization
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, 
                 num_features=256, ortho_features=True, redraw=False):
        """
        Initialize the Performer encoder layer.
        
        Args:
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            dropout (float): Dropout rate
            num_features (int): Number of random features for approximation
            ortho_features (bool): Whether to use orthogonal random features
            redraw (bool): Whether to redraw random features for each forward pass
        """
        super().__init__()
        # STEP 18: Initialize multi-head attention
        # Create PerformerMultiHeadAttention instance
        self.attention = PerformerMultiHeadAttention(
            d_model, 
            num_heads, 
            num_features=num_features, 
            ortho_features=ortho_features, 
            redraw=redraw
        )
        
        # STEP 19: Initialize feed-forward network
        # Create a two-layer MLP with ReLU activation
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # STEP 20: Initialize layer normalizations
        # Two LayerNorm instances: attention and feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # STEP 21: Initialize dropout
        # Create dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Pass the input through the encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 22: Apply attention sub-layer
        # Multi-head attention -> Dropout -> Add & Norm
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # STEP 23: Apply feed-forward sub-layer
        # Feed-forward -> Dropout -> Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Performer(nn.Module):
    """
    Full Performer model for sequence modeling.
    """
    
    def __init__(self, d_model, num_layers, num_heads, d_ff, max_seq_length, 
                 num_features=256, ortho_features=True, redraw=False, dropout=0.1):
        """
        Initialize the Performer model.
        
        Args:
            d_model (int): Model's dimension
            num_layers (int): Number of encoder layers
            num_heads (int): Number of attention heads
            d_ff (int): Hidden dimension of feed-forward network
            max_seq_length (int): Maximum sequence length
            num_features (int): Number of random features for approximation
            ortho_features (bool): Whether to use orthogonal random features
            redraw (bool): Whether to redraw random features for each forward pass
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 24: Initialize positional encoding
        # Create positional embeddings
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # STEP 25: Initialize encoder layers
        # Create a ModuleList of PerformerEncoderLayer instances
        self.layers = nn.ModuleList([
            PerformerEncoderLayer(
                d_model, 
                num_heads, 
                d_ff, 
                dropout=dropout,
                num_features=num_features,
                ortho_features=ortho_features,
                redraw=redraw
            )
            for _ in range(num_layers)
        ])
        
        # STEP 26: Initialize final layer normalization
        # Create LayerNorm for the output
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Pass the input through the Performer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 27: Add positional encoding
        # Add positional information to the input
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # STEP 28: Pass through encoder layers
        # Process through each encoder layer in sequence
        for layer in self.layers:
            x = layer(x)
        
        # STEP 29: Apply final layer normalization
        # Normalize the output
        x = self.norm(x)
        
        return x