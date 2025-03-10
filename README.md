# Transformer From Scratch

A step-by-step tutorial for implementing Transformer architectures from scratch in PyTorch.

## Overview

This repository provides detailed, well-documented implementations of Transformer architectures. Each implementation includes:

- 📝 Step-by-step explanations
- 💡 Detailed comments for every key component
- 🔍 Mathematical derivations where necessary
- ⚡ Working examples and usage demonstrations

## Structure

```
transformer_from_scratch/
├── basics/
│   ├── attention.py          # Basic attention mechanism explained
│   ├── positional_enc.py     # Positional encoding implementation
│   └── feed_forward.py       # Feed-forward network details
├── nlp/
│   ├── vanilla/
│   │   ├── encoder.py        # Classic Transformer encoder
│   │   ├── decoder.py        # Classic Transformer decoder
│   │   └── full_model.py     # Complete Transformer model
│   └── modern/
│       ├── bert/            # BERT implementation details
│       └── gpt/             # GPT model structure
└── vision/
    ├── vit/
    │   ├── patch_embed.py    # Image patching and embedding
    │   └── vit_model.py      # Full ViT implementation
    └── modern/
        └── ...              # Modern vision transformer variants
```

## How to Use

Each implementation file serves as both a tutorial and a working module. For example, to understand the basic attention mechanism:

```python
# attention.py contains detailed explanations like:

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.
    
    Paper: "Attention Is All You Need" - https://arxiv.org/abs/1706.03762
    
    Step 1: Calculate attention scores
        - Multiply query with key (matrix multiplication)
        - Scale by sqrt(d_k)
    Step 2: Apply softmax to get attention weights
    Step 3: Multiply with values to get weighted sum
    """
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, mask=None):
        # Detailed comments explaining each step...
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/xfey/transformer-from-scratch.git
cd transformer-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the basics:
   - Read through `basics/attention.py` for attention mechanism
   - Move on to `nlp/vanilla/encoder.py` for full encoder implementation
   - Each file contains detailed explanations and comments

## Learning Path

1. **Fundamentals**
   - Attention mechanism
   - Positional encoding
   - Feed-forward networks

2. **Classic Transformer**
   - Encoder implementation
   - Decoder implementation
   - Full model assembly

3. **Modern Variants**
   - BERT architecture
   - GPT models
   - Vision Transformers

## Contributing

Contributions are welcome! Please feel free to:
- Add new implementations
- Improve existing explanations
- Fix bugs or add clarifications
- Suggest new learning paths

## License

This project is licensed under the MIT License - see the LICENSE file for details.
