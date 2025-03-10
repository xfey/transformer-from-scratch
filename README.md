# Transformer From Scratch

A step-by-step tutorial for implementing Transformer architectures from scratch in PyTorch.


## 如何开始

1. 阅读代码，理解模型结构的理论实现
2. 根据代码中的提示，尝试实现模型结构
3. 对比 `solutions` 目录下的代码，检查代码是否正确

## Getting Started

1. Read the code for each architecture, understand its theoretical basis.
2. Try to implement the code with the hint in the code.
3. Compare your implementation with the provided solution.


## Overview

This repository provides detailed, well-documented implementations of Transformer architectures. Each implementation includes:

- 📝 Step-by-step explanations
- 💡 Detailed comments for every key component
- 🔍 Mathematical derivations where necessary
- ⚡ Working examples and demonstrations

## Structure

```
transformer_from_scratch/
├── vanilla_transformer/          # 经典 Transformer 结构
│   ├── attention.py             # 注意力机制实现
│   ├── encoder.py              # 编码器实现
│   ├── decoder.py              # 解码器实现
│   └── transformer.py          # 完整模型
│
├── attention_variants/          # 注意力机制变体
│   ├── linear_attention/       # 线性注意力
│   ├── sparse_attention/       # 稀疏注意力
│   └── efficient_attention/    # 高效注意力
│
├── efficient_transformers/      # 轻量级/高效架构
│   ├── performer/              # Performer
│   ├── reformer/              # Reformer
│   ├── efficient_vit/         # EfficientViT
│   └── fast_vit/              # FastViT
│
├── vision_transformers/        # 视觉 Transformer
│   ├── vit/                   # Vision Transformer
│   ├── swin/                  # Swin Transformer
│   └── modern_variants/       # 新型视觉 Transformer
│
├── specialized/               # 特定任务架构
│   ├── detection/            # 目标检测
│   └── generation/          # 生成模型
│
└── solutions/                # 答案
```

## Implemented Architectures

| Architecture | Paper | Original Repo | Implementation Path |
|-------------|-------|---------------|-------------------|
| Vanilla Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) | `vanilla_transformer/` |
| Linear Attention | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | [idiap/fast-transformers](https://github.com/idiap/fast-transformers) | `attention_variants/linear_attention/` |
| Performer | [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) | [google-research/performer](https://github.com/google-research/performer) | `efficient_transformers/performer/` |
| Vision Transformer | [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | [google-research/vision_transformer](https://github.com/google-research/vision_transformer) | `vision_transformers/vit/` |
| Swin Transformer | [Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030) | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | `vision_transformers/swin/` |
| EfficientViT | [EfficientViT](https://arxiv.org/abs/2205.14756) | [microsoft/EfficientViT](https://github.com/microsoft/EfficientViT) | `efficient_transformers/efficient_vit/` |
| DETR | [End-to-End Object Detection](https://arxiv.org/abs/2005.12872) | [facebookresearch/detr](https://github.com/facebookresearch/detr) | `specialized/detection/` |


## Contributing

Contributions are welcome! Please feel free to:
- Add new implementations
- Improve existing explanations
- Suggest new architectures

## License

This project is licensed under the MIT License - see the LICENSE file for details.
