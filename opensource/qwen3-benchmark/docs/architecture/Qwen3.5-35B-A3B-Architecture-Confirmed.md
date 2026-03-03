# Qwen3.5-35B-A3B 架构确认报告

## 1. 关键发现

已从 HuggingFace 获取 Qwen3.5-35B-A3B 的 `config.json`，架构与 Qwen3-Next 有**显著差异**。

**配置来源**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## 2. 核心参数对比

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B (实际) | 差异分析 |
|------|-------------------|---------------------|---------|
| **架构类型** | Hybrid Mamba-Transformer-MoE | **Linear Attention + Full Attention + MoE** | 完全不同 |
| **总层数** | 96 | **40** | -58% |
| **Hidden Size** | 2048 | **2048** | 相同 |
| **注意力头** | 16 | **16** | 相同 |
| **GQA头** | 2 | **2** | 相同 |
| **专家数** | 512 | **256** | -50% |
| **TopK** | 10 | **8** | 减少 |
| **专家维度** | 512 | **512** | 相同 |
| **共享专家** | 512 | **512** | 相同 |
| **最大长度** | 262144 | **262144** | 相同 |
| **激活函数** | SwiGLU | **SiLU** | 可能兼容 |
| **词表大小** | 151936 | **248320** | +63% |
| **Layer Pattern** | M-M-M-* (Mamba) | **L-L-L-F** (Linear Attn) | 完全不同 |

## 3. 关键架构特性

### 3.1 Linear Attention (替代 Mamba)

```json
"layer_types": [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ... (共40层，每4层一个循环)
]
```

**周期模式**: 3层 Linear Attention + 1层 Full Attention

### 3.2 MLA-like 参数

Qwen3.5 使用类 MLA (Multi-Head Latent Attention) 结构：

```json
"head_dim": 256,
"linear_conv_kernel_dim": 4,
"linear_key_head_dim": 128,
"linear_num_key_heads": 16,
"linear_num_value_heads": 32,
"linear_value_head_dim": 128,
```

| 参数 | 值 | 说明 |
|------|-----|------|
| `linear_conv_kernel_dim` | 4 | 线性注意力卷积核维度 |
| `linear_key_head_dim` | 128 | Key 头维度 |
| `linear_value_head_dim` | 128 | Value 头维度 |
| `linear_num_key_heads` | 16 | Key 头数 |
| `linear_num_value_heads` | 32 | Value 头数 |

### 3.3 Multi-Token Prediction (MTP)

```json
"mtp_num_hidden_layers": 1,
"mtp_use_dedicated_embeddings": false,
```

- 启用 MTP 训练
- 使用共享 embedding

### 3.4 多模态支持

```json
"architectures": ["Qwen3_5MoeForConditionalGeneration"],
"image_token_id": 248056,
"video_token_id": 248057,
```

包含独立的 Vision Config：
- Depth: 27
- Hidden Size: 1152
- Patch Size: 16

### 3.5 RoPE 配置

```json
"rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25
}
```

- **MRoPE** (Multi-modal RoPE) 支持
- 部分旋转因子: 0.25 (与 qwen3_next 相同)
- Theta: 10000000 (相同)

## 4. PAI-Megatron-Patch 支持评估

### 4.1 ✅ 已支持

| 特性 | 支持状态 | 说明 |
|------|---------|------|
| **MoE (256专家, topk=8)** | ✅ 支持 | 标准 Megatron MoE |
| **GQA (2头)** | ✅ 支持 | `--group-query-attention --num-query-groups 2` |
| **SiLU激活** | ✅ 支持 | `--swiglu` (可能需要调整) |
| **RMSNorm (1e-6)** | ✅ 支持 | `--norm-epsilon 1e-6` |
| **RoPE (theta=1e7)** | ✅ 支持 | `--rotary-base 10000000` |
| **partial_rotary_factor=0.25** | ✅ 支持 | `--rotary-percent 0.25` |
| **MTP** | ⚠️ 部分支持 | Megatron-LM-250908+ 支持 |

### 4.2 ⚠️ 需验证

| 特性 | 风险等级 | 说明 |
|------|---------|------|
| **Linear Attention** | 🔴 **高风险** | 非标准 Mamba，需确认实现 |
| **MLA-like结构** | 🟡 中风险 | 参数与 DeepSeek MLA 类似但命名不同 |
| **MRoPE** | 🟡 中风险 | 多模态 RoPE，需验证支持 |
| **Vocab Size 248320** | 🟢 低风险 | 仅需调整 `--padded-vocab-size` |

### 4.3 🔴 高风险项详细分析

#### Linear Attention (vs Mamba)

Qwen3.5 使用 `linear_attention` 而非 Qwen3-Next 的 `mamba`：

```python
# Qwen3.5 配置
"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]

# Qwen3-Next 配置
"hybrid_override_pattern": "M-M-M-*"  # Mamba-Mamba-Mamba-Attention
```

**关键区别**:
- Linear Attention 可能是 **GLA (Gated Linear Attention)** 或类似变体
- 与 Mamba 不同，但都属于线性复杂度注意力
- PAI-Megatron-Patch 的 `qwen3_next` 实现**不能直接复用**

#### MLA-like 结构

Qwen3.5 的 Linear Attention 有 MLA 风格的维度设置：
- `linear_key_head_dim=128`
- `linear_value_head_dim=128`
- `linear_num_key_heads=16`
- `linear_num_value_heads=32`

这与 DeepSeek-V2 的 MLA 类似，但参数名不同，需要映射到 Megatron 的 MLA 参数：
```bash
--kv-lora-rank ?
--qk-head-dim ?
--v-head-dim ?
```

## 5. 适配建议（更新）

### 5.1 启动脚本关键参数

```bash
# 基础参数
NUM_LAYERS=40
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=2
INTERMEDIATE_SIZE=8192  # 估算值，需确认
MOE_INTERMEDIATE_SIZE=512
MAX_POSITION_EMBEDDINGS=262144
VOCAB_SIZE=248320  # 注意：比Qwen3-Next大很多
ROPE_THETA=10000000
RMS_NORM_EPS=1e-6

# MoE参数
NUM_EXPERTS=256
ROUTER_TOPK=8
SHARED_EXPERT_INTERMEDIATE_SIZE=512

# 新增：MTP
MTP_NUM_LAYERS=1

# 新增：Linear Attention（需确认PAI支持情况）
LINEAR_ATTENTION_CONFIG="..."  # 待确定
```

### 5.2 Layer Pattern 映射

Qwen3.5 的 40 层分布：
- 30层 Linear Attention (75%)
- 10层 Full Attention (25%)

需要验证 PAI-Megatron-Patch 是否支持这种混合模式。

### 5.3 词表处理

```bash
--padded-vocab-size 248320
--patch-tokenizer-type Qwen3Tokenizer  # 可能需要更新tokenizer
```

## 6. 建议的验证步骤

1. **确认 Linear Attention 实现**
   - 检查 PAI-Megatron-Patch 是否有 GLA 或类似实现
   - 如果没有，需要评估是否可以使用 Full Attention 作为 fallback

2. **MLA参数映射**
   - 将 `linear_key_head_dim` 等映射到 Megatron 的 MLA 参数
   - 验证维度计算是否正确

3. **MTP训练**
   - 确认 `--mtp-num-layers 1` 支持
   - 验证 MTP 与 MoE 的兼容性

4. **权重转换**
   - 由于架构差异大，转换工具可能需要大幅修改
   - 特别是 Linear Attention 层的权重映射

## 7. 结论

Qwen3.5-35B-A3B 的架构比预期更复杂：

1. **不是简单的缩放版**，而是完全不同的架构（Linear Attention vs Mamba）
2. **引入 MLA-like 结构**，需要仔细验证支持情况
3. **词表扩大 63%**，需要调整 embedding 层
4. **MTP 训练** 需要额外支持

**适配难度**: 从 "低" 提升到 **"中高"**，建议等待 PAI-Megatron-Patch 官方支持或深入验证 Linear Attention 实现。

---

**报告更新时间**: 2026-03-02  
**基于配置**: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
