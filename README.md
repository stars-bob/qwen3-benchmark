# Qwen3 Benchmark Repository

This repository contains adaptation guides and research documents for training Qwen3 series models (including Qwen3.5) using PAI-Megatron-Patch framework.

## Repository Structure

```
qwen3-benchmark/
├── README.md                                      # This file
├── Qwen3.5-35B-A3B-Adaptation-Guide.md           # Adaptation guide for Qwen3.5-35B-A3B
└── Qwen3-Next-vs-Qwen3.5-Architecture-Research.md # Architecture comparison research
```

## Documents

### 1. Qwen3.5-35B-A3B Adaptation Guide

A practical guide for adapting Qwen3.5-35B-A3B model to PAI-Megatron-Patch without modifying the original repository.

**Key Topics:**
- Parameter configuration based on config.json
- Custom launch script creation
- Weight conversion workflow
- Training optimization recommendations

### 2. Qwen3-Next vs Qwen3.5 Architecture Research

Deep analysis of architectural differences between Qwen3-Next-80B-A3B and Qwen3.5-35B-A3B.

**Key Findings:**
- Hybrid Mamba-Transformer-MoE architecture details
- Parameter scaling patterns
- Risk assessment for adaptation
- Recommended adaptation strategy

## Quick Start

### Prerequisites

- PAI-Megatron-Patch v0.12.3+
- Megatron-LM-250908 backend
- 8x NVIDIA H20 (80GB) or equivalent

### Usage

1. Follow the [Adaptation Guide](./Qwen3.5-35B-A3B-Adaptation-Guide.md) to set up training
2. Refer to [Architecture Research](./Qwen3-Next-vs-Qwen3.5-Architecture-Research.md) for technical details

## Related Resources

- [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen3)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## License

This documentation follows the same license as PAI-Megatron-Patch (Apache 2.0).

## Contributing

This repository contains research and adaptation documentation. For code contributions, please refer to the [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) main repository.

---

**Last Updated**: 2026-03-02
