"""
Complete Transformer Implementation
--------------------------------

The Transformer model consists of:
1. Input/Output Embeddings
2. Positional Encoding
3. Encoder Stack
4. Decoder Stack
5. Final Linear Layer and Softmax

Key Components:
-------------
- Token Embeddings: Convert input tokens to vectors
- Positional Encoding: Add position information
- Encoder: Process input sequence
- Decoder: Generate output sequence
- Output Layer: Convert to probability distribution

Dimensions Guide:
---------------
- Batch size: B
- Source sequence length: S
- Target sequence length: T
- Model dimension: d_model
- Vocabulary size: vocab_size
- Number of encoder/decoder layers: N
- Number of attention heads: h
- Feed-forward hidden dimension: d_ff

Shape transformations:
- Input tokens: (B, S) -> Embeddings: (B, S, d_model)
- Encoder output: (B, S, d_model)
- Decoder output: (B, T, d_model)
- Final output: (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module.
    
    Adds positional information to input embeddings:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_seq_length=5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Model's dimension
            max_seq_length (int): Maximum sequence length
        """
        super().__init__()
        # STEP 1: Create positional encoding matrix
        # Shape: (max_seq_length, d_model)
        #   - create position with `torch.arange`
        #   - create div_term following the formula
        #   - create the pe tensor
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # STEP 2: Register buffer (not a parameter, but part of the state)
        # This ensures the encoding moves with the model between CPU/GPU
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, L, d_model)
        """
        # STEP 3: Add positional encoding to input
        # Select the appropriate length of encoding and add to input
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    """
    Complete Transformer model.
    """
    
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_layers=6,
                 d_ff=2048,
                 max_seq_length=5000,
                 dropout=0.1):
        """
        Initialize the Transformer.
        
        Args:
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
            d_model (int): Model's dimension
            num_heads (int): Number of attention heads
            num_layers (int): Number of encoder/decoder layers
            d_ff (int): Feed-forward hidden dimension
            max_seq_length (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super().__init__()
        # STEP 4: Create embedding layers
        # Source and target embeddings with appropriate vocab sizes
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # STEP 5: Create positional encoding layer
        # Initialize PositionalEncoding module
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # STEP 6: Create encoder
        # Initialize TransformerEncoder
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # STEP 7: Create decoder
        # Initialize TransformerDecoder
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # STEP 8: Create output layer
        # Linear layer to project to vocabulary size
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # STEP 9: Initialize parameters
        # Xavier/Glorot initialization for better training
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def create_masks(self, src, tgt):
        """
        Create source and target masks.
        
        Args:
            src (torch.Tensor): Source tensor of shape (B, S)
            tgt (torch.Tensor): Target tensor of shape (B, T)
            
        Returns:
            tuple: (src_mask, tgt_mask)
                - src_mask shape: (B, 1, 1, S)
                - tgt_mask shape: (B, 1, T, T)
        """
        # STEP 10: Create source mask
        # Mask pad tokens in source sequence
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # STEP 11: Create target mask
        # Combine padding mask and subsequent mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.triu(torch.ones((tgt_len, tgt_len)), diagonal=1).bool()
        tgt_mask = tgt_pad_mask & ~tgt_sub_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        """
        Forward pass of the model.
        
        Args:
            src (torch.Tensor): Source tensor of shape (B, S)
            tgt (torch.Tensor): Target tensor of shape (B, T)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, tgt_vocab_size)
        """
        # STEP 12: Create masks
        # Get source and target masks
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # STEP 13: Process source input
        # Embedding -> Positional Encoding -> Encoder
        src_embedded = self.positional_encoding(self.src_embed(src) * math.sqrt(self.d_model))
        enc_output = self.encoder(src_embedded, src_mask)
        
        # STEP 14: Process target input
        # Embedding -> Positional Encoding -> Decoder
        tgt_embedded = self.positional_encoding(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # STEP 15: Generate final output
        # Project to vocabulary size and apply log_softmax
        output = F.log_softmax(self.output_layer(dec_output), dim=-1)
        
        return output 