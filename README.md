# Qwen3 Benchmark Repository

This repository contains adaptation guides and research documents for training Qwen3 series models (including Qwen3.5) using PAI-Megatron-Patch framework.

## ⚠️ Important Update (2026-03-02)

**Architecture Correction**: Both Qwen3-Next and Qwen3.5 use **Linear Attention** (not Mamba as previously thought). This significantly reduces the adaptation difficulty for Qwen3.5.

## Repository Structure

```
qwen3-benchmark/
├── README.md                                      # This file
├── Qwen3.5-35B-A3B-Adaptation-Guide.md           # Practical adaptation guide for Qwen3.5
├── Qwen3-Next-vs-Qwen3.5-Architecture-Research.md # Architecture comparison (corrected)
└── HF-Config-Comparison-Qwen3.5-vs-Qwen3-Next.md  # Raw HF config comparison
```

## Documents

### 1. Qwen3.5-35B-A3B Adaptation Guide

A practical guide for adapting Qwen3.5-35B-A3B model to PAI-Megatron-Patch without modifying the original repository.

**Key Topics:**
- Parameter configuration based on HuggingFace config.json
- Custom launch script creation
- Weight conversion workflow with vocab_size=248320
- MTP (Multi-Token Prediction) support
- Training optimization recommendations

### 2. Qwen3-Next vs Qwen3.5 Architecture Research (Corrected)

Deep analysis of architectural differences between Qwen3-Next-80B-A3B and Qwen3.5-35B-A3B.

**Key Findings:**
- ✅ Both use **Linear Attention** (not Mamba)
- Architecture similarity: **95%**
- Main differences: layers (48→40), experts (512→256), vocab (151936→248320)
- Qwen3.5 adds MTP support
- Adaptation difficulty: **Medium-Low** (previously assessed as High)

### 3. HF Config Comparison

Raw side-by-side comparison of HuggingFace configuration files.

## Quick Comparison

| Feature | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B |
|---------|-------------------|-----------------|
| **Architecture** | Linear Attention + MoE | Linear Attention + MoE |
| **Layers (HF)** | 48 | **40** |
| **Experts** | 512 | **256** |
| **TopK** | 10 | **8** |
| **Vocab Size** | 151936 | **248320** (+63%) |
| **MTP** | ❌ No | ✅ Yes |
| **Multi-modal** | ❌ No | ✅ Yes |

## Quick Start

### Prerequisites

- PAI-Megatron-Patch v0.12.3+
- Megatron-LM-250908 backend
- 8x NVIDIA H20 (80GB) or equivalent

### Usage

1. Follow the [Adaptation Guide](./Qwen3.5-35B-A3B-Adaptation-Guide.md) to set up training
2. Refer to [Architecture Research](./Qwen3-Next-vs-Qwen3.5-Architecture-Research.md) for technical details
3. Check [HF Config Comparison](./HF-Config-Comparison-Qwen3.5-vs-Qwen3-Next.md) for raw configuration data

## Configuration Sources

- **Qwen3-Next-80B-A3B**: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- **Qwen3.5-35B-A3B**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## Related Resources

- [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen3)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

## Changelog

### 2026-03-02
- **Major Correction**: Both models use Linear Attention, not Mamba
- Updated adaptation difficulty from "High" to "Medium-Low"
- Added MTP support documentation
- Updated vocab_size handling (248320)

## License

This documentation follows the same license as PAI-Megatron-Patch (Apache 2.0).

## Contributing

This repository contains research and adaptation documentation. For code contributions, please refer to the [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) main repository.

---

**Last Updated**: 2026-03-02  
**Status**: Architecture corrected based on official HF configs
