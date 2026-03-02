# Qwen3 Benchmark Repository

Qwen3 系列模型的基准测试与训练适配资源汇总。

---

## 📁 目录

- [Part 1: Apple Silicon 推理测试](#part-1-apple-silicon-推理测试) - Qwen3-1.7B PyTorch vs MLX 对比
- [Part 2: PAI-Megatron-Patch 训练适配](#part-2-pai-megatron-patch-训练适配) - Qwen3.5-35B-A3B 适配指南
- [Part 3: 架构调研报告](#part-3-架构调研报告) - Qwen3-Next vs Qwen3.5 深度分析

---

## Part 1: Apple Silicon 推理测试

Apple Silicon 上的 Qwen3-1.7B 模型推理对比测试。

### 📊 测试内容

| 脚本 | 说明 |
|------|------|
| `test_qwen3.py` | PyTorch + MPS 版本推理 |
| `test_qwen3_mlx.py` | MLX bf16 版本推理 |
| `compare_precision.py` | 输出质量对比测试 |
| `compare_ppl.py` | Perplexity 数值对比 |

### 🎯 核心发现

#### 速度对比

| 指标 | PyTorch FP16 | MLX bf16 | 提升 |
|------|-------------|----------|------|
| 模型加载 | ~120s | ~2s | **60x** |
| 推理速度 | 35.52s | 23.97s | **1.48x** |

#### 精度对比

- **输出质量**：几乎完全一致（思维链、回答内容相同）
- **PPL 差异**：仅 1.38%（在可接受范围内）

#### 推荐

在 Apple Silicon (M4 Pro) 上 **优先使用 MLX bf16 版本**：
- ✅ 加载速度碾压级优势
- ✅ 推理速度快 48%
- ✅ 输出质量无损失
- ✅ 内存效率更高

### 🚀 使用方法

#### 环境准备

```bash
# 创建 conda 环境
conda create -n transformers python=3.12
conda activate transformers

# 安装依赖
pip install torch torchvision torchaudio transformers accelerate
pip install mlx-lm
```

#### 运行测试

```bash
# PyTorch 版本
python test_qwen3.py

# MLX 版本（推荐）
python test_qwen3_mlx.py

# 对比测试
python compare_precision.py
python compare_ppl.py
```

### 🔧 硬件要求

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- 8GB+ 统一内存（推荐 16GB+）

---

## Part 2: PAI-Megatron-Patch 训练适配

### 文档

- **[Qwen3.5-35B-A3B-Adaptation-Guide.md](./Qwen3.5-35B-A3B-Adaptation-Guide.md)** - Qwen3.5-35B-A3B 适配指南

本部分提供在不改动原仓库的前提下，将 Qwen3.5-35B-A3B 模型适配到 PAI-Megatron-Patch 框架的训练建议。

### 核心内容

- 参数配置（基于 config.json）
- 自定义启动脚本创建
- 权重转换工作流
- 训练优化建议

### 快速开始

#### 环境要求

- PAI-Megatron-Patch v0.12.3+
- Megatron-LM-250908 backend
- 8x NVIDIA H20 (80GB) 或等效配置

#### 启动训练

```bash
# 1. 下载模型
modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir ./Qwen3.5-35B-A3B

# 2. 转换权重
bash convert_qwen3.5_35b.sh bf16

# 3. 启动训练
bash run_qwen3.5_35b.sh dlc 35B-A3B 1 8 1e-5 1e-6 4096 4096 bf16 2 2 1 1 4 \
    true true false false sel false 10000 ./data ./data ./ckpts 100000000 10000 ./output
```

---

## Part 3: 架构调研报告

### 文档

- **[Qwen3-Next-vs-Qwen3.5-Architecture-Research.md](./Qwen3-Next-vs-Qwen3.5-Architecture-Research.md)** - 架构对比分析

深入分析 PAI-Megatron-Patch 框架中已支持的 Qwen3-Next-80B-A3B 与目标模型 Qwen3.5-35B-A3B 之间的架构差异。

### 关键发现

| 维度 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B (推测) |
|------|-------------------|----------------------|
| 总参数量 | 80B | 35B (-56%) |
| 层数 | 96 | 48-64 (估算) |
| 专家数 | 512 | 128-256 (估算) |
| 架构 | Hybrid Mamba-Transformer-MoE | 预计相同 |

### 适配风险评估

- ✅ **低风险**：Tokenizer、Mamba 实现、MoE 路由可直接复用
- ⚠️ **中风险**：Hybrid Pattern 长度需匹配层数
- ❓ **待确认**：需获取官方 config.json 验证参数

---

## 📂 仓库结构

```
qwen3-benchmark/
├── README.md                                      # 本文件
├── Qwen3.5-35B-A3B-Adaptation-Guide.md           # Qwen3.5-35B-A3B 适配指南
├── Qwen3-Next-vs-Qwen3.5-Architecture-Research.md # 架构对比调研
│
├── test_qwen3.py                                  # PyTorch 推理脚本
├── test_qwen3_mlx.py                              # MLX 推理脚本
├── compare_precision.py                           # 精度对比
└── compare_ppl.py                                 # PPL 对比
```

---

## 🔗 相关资源

- [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)
- [Qwen3 Official Repository](https://github.com/QwenLM/Qwen3)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

---

## 📄 许可

- 推理测试部分：MIT License
- 训练适配部分：Apache 2.0（与 PAI-Megatron-Patch 一致）

---

**Last Updated**: 2026-03-02
