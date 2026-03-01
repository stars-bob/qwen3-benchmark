# Qwen3-1.7B Benchmark

Apple Silicon 上的 Qwen3-1.7B 模型推理对比测试。

## 📊 测试内容

| 脚本 | 说明 |
|------|------|
| `test_qwen3.py` | PyTorch + MPS 版本推理 |
| `test_qwen3_mlx.py` | MLX bf16 版本推理 |
| `compare_precision.py` | 输出质量对比测试 |
| `compare_ppl.py` | Perplexity 数值对比 |

## 🎯 核心发现

### 速度对比

| 指标 | PyTorch FP16 | MLX bf16 | 提升 |
|------|-------------|----------|------|
| 模型加载 | ~120s | ~2s | **60x** |
| 推理速度 | 35.52s | 23.97s | **1.48x** |

### 精度对比

- **输出质量**：几乎完全一致（思维链、回答内容相同）
- **PPL 差异**：仅 1.38%（在可接受范围内）

### 推荐

在 Apple Silicon (M4 Pro) 上 **优先使用 MLX bf16 版本**：
- ✅ 加载速度碾压级优势
- ✅ 推理速度快 48%
- ✅ 输出质量无损失
- ✅ 内存效率更高

## 🚀 使用方法

### 环境准备

```bash
# 创建 conda 环境
conda create -n transformers python=3.12
conda activate transformers

# 安装依赖
pip install torch torchvision torchaudio transformers accelerate
pip install mlx-lm
```

### 运行测试

```bash
# PyTorch 版本
python test_qwen3.py

# MLX 版本（推荐）
python test_qwen3_mlx.py

# 对比测试
python compare_precision.py
python compare_ppl.py
```

## 📁 文件说明

```
qwen3-benchmark/
├── test_qwen3.py           # PyTorch FP16 推理脚本
├── test_qwen3_mlx.py       # MLX bf16 推理脚本
├── compare_precision.py    # 输出质量对比
├── compare_ppl.py          # PPL 数值对比
└── README.md               # 本文件
```

## 🔧 硬件要求

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15.0+
- 8GB+ 统一内存（推荐 16GB+）

## 📄 许可

MIT License
