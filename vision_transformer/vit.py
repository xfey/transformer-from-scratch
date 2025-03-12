"""
Vision Transformer (ViT) Implementation
--------------------------------------

The Vision Transformer (ViT) applies the Transformer architecture to image classification.
Instead of processing sequences of tokens, ViT splits an image into fixed-size patches,
linearly embeds each patch, adds position embeddings, and feeds the resulting sequence
of vectors to a standard Transformer encoder.

Key Concepts:
------------
1. Patch Embedding:
   - Split image into non-overlapping patches
   - Flatten and linearly project each patch to obtain patch embeddings
   
2. Position Embedding:
   - Add learnable position embeddings to provide spatial information
   
3. Class Token:
   - Prepend a learnable [CLS] token to the sequence
   - The final state of this token is used for classification

4. Transformer Encoder:
   - Process the sequence of patch embeddings using self-attention

Dimensions Guide:
---------------
- Batch size: B
- Image size: (H, W, C) - Height, Width, Channels
- Patch size: (P, P) - Height and Width of each patch
- Number of patches: N = (H*W)/(P*P)
- Embedding dimension: D
- Number of classes: num_classes

Shape transformations:
- Input image: (B, C, H, W)
- Patches: (B, N, P*P*C)
- Patch embeddings: (B, N, D)
- With class token: (B, N+1, D)
- Transformer output: (B, N+1, D)
- Classification output: (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Steps:
    1. Split image into non-overlapping patches
    2. Flatten each patch
    3. Project to embedding dimension
    """
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        """
        Initialize the patch embedding layer.
        
        Args:
            img_size (int): Size of the input image (assumed to be square)
            patch_size (int): Size of each patch (assumed to be square)
            in_channels (int): Number of input channels
            embed_dim (int): Dimension of the patch embeddings
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # STEP 1: Calculate number of patches
        # Formula: num_patches = (img_size / patch_size)^2
        
        
        # STEP 2: Create projection layer
        # Use a Conv2d layer with kernel_size=patch_size and stride=patch_size
        # This effectively splits the image into patches and projects them

        
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
        # STEP 3: Apply the projection layer
        # This splits the image into patches and projects them
        # target shape: (B, embed_dim, H/patch_size, W/patch_size)

        
        # STEP 4: Reshape the output to (B, num_patches, embed_dim)
        # Hint: flatten -> (B, embed_dim, num_patches)
        #       transpose -> (B, num_patches, embed_dim)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # STEP 1: Create self-attention block: layer normalization + self-attention
        #         hint: use nn.MultiheadAttention()

        
        # STEP 2: Create MLP block: linear + gelu + dropout + linear + dropout
        #         with a layer normalization at the beginning
        #         hint: mlp_hidden_dim = int(embed_dim * mlp_ratio)


    def forward(self, x):
        # Self-attention block

        
        # MLP block


        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple self-attention layers.
    """
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # STEP 1: Create encoder layers
        # Create a ModuleList of EncoderLayer instances

        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768,
                 num_layers=12, num_heads=12, mlp_ratio=4.0, dropout=0.1, num_classes=1000):
        """
        Initialize the Vision Transformer.
        
        Args:
            img_size (int): Input image size (assumed to be square)
            patch_size (int): Patch size (assumed to be square)
            in_channels (int): Number of input channels
            embed_dim (int): Embedding dimension
            num_layers (int): Number of Transformer layers
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
            dropout (float): Dropout rate
            num_classes (int): Number of classes for classification
        """
        super().__init__()
        
        # STEP 5: Create patch embedding layer
        # Use the PatchEmbedding class
        # Then, save the num_patches
        
        
        # STEP 6: Create class token
        # This is a learnable embedding prepended to the patch embeddings
        
        
        # STEP 7: Create position embeddings
        # These are added to the patch embeddings to retain positional information
        
        
        # Initialize class token and position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        
        # STEP 8: Create embedding dropout
        # Applied after adding position embeddings
        
        
        # STEP 9: Create Transformer encoder
        # Use nn.TransformerEncoder

        
        # STEP 10: Create layer normalization
        # Applied before the final classification head
        
        
        # STEP 11: Create classification head
        # Projects from embed_dim to num_classes
        
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        # STEP 12: Get patch embeddings
        # Use the patch embedding layer
        
        
        # STEP 13: Add class token
        # Expand class token to batch size and prepend to patch embeddings
        # Hint: get batch_size first, then expand cls_token
        
        
        # STEP 14: Add position embeddings
        # Add position embeddings to the combined [class] and patch tokens
        
        
        # STEP 15: Apply embedding dropout
        # Use the dropout layer
        
        
        # STEP 16: Pass through Transformer encoder
        # Apply the Transformer encoder to the sequence
        
        
        # STEP 17: Apply layer normalization
        # Use the layer norm
        
        
        # STEP 18: Extract class token
        # The class token is the first token in the sequence
        
        
        # STEP 19: Apply classification head
        # Project to num_classes
        
        
        return x 