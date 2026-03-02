# Qwen3-Next vs Qwen3.5-35B-A3B 架构调研报告（已修正）

> **修正说明**: 本报告已根据 HuggingFace 官方配置更新，纠正了之前关于 "Mamba" 架构的错误认知。

## 1. 调研背景

本文档基于 HuggingFace 官方配置，深入分析 Qwen3-Next-80B-A3B 与 Qwen3.5-35B-A3B 的真实架构差异。

**配置来源**:
- Qwen3-Next-80B-A3B: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

## 2. HF 配置直接对比

### 2.1 基础架构参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **num_hidden_layers** | 48 | **40** | -8 (-17%) |
| **hidden_size** | 2048 | **2048** | = |
| **num_attention_heads** | 16 | **16** | = |
| **num_key_value_heads** | 2 | **2** | = |
| **head_dim** | 256 | **256** | = |
| **intermediate_size** | 5120 | **?** (text_config 未直接显示) | 待确认 |
| **moe_intermediate_size** | 512 | **512** | = |
| **hidden_act** | "silu" | **"silu"** | = |
| **vocab_size** | 151936 | **248320** | **+63%** ⚠️ |
| **max_position_embeddings** | 262144 | **262144** | = |

### 2.2 MoE 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **num_experts** | 512 | **256** | -256 (-50%) |
| **num_experts_per_tok** | 10 | **8** | -2 |
| **shared_expert_intermediate_size** | 512 | **512** | = |
| **router_aux_loss_coef** | 0.001 | **0.001** | = |

### 2.3 Linear Attention 参数 (关键发现!)

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **linear_conv_kernel_dim** | 4 | **4** | = |
| **linear_key_head_dim** | 128 | **128** | = |
| **linear_num_key_heads** | 16 | **16** | = |
| **linear_num_value_heads** | 32 | **32** | = |
| **linear_value_head_dim** | 128 | **128** | = |
| **full_attention_interval** | 4 | **4** | = |

**重要发现**: 两者都使用 Linear Attention，架构高度相似！

### 2.4 RoPE 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **rope_theta** | 10000000 | **10000000** | = |
| **partial_rotary_factor** | 0.25 | **0.25** | = |
| **rope_type** | "default" | **"default"** | = |

Qwen3.5 新增 MRoPE 支持（多模态）：
```json
"rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10]
}
```

### 2.5 归一化参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **rms_norm_eps** | 1e-06 | **1e-06** | = |

### 2.6 新增/移除参数

**Qwen3.5 新增**:
```json
"mtp_num_hidden_layers": 1              // Multi-Token Prediction
"mtp_use_dedicated_embeddings": false
"vision_config": {...}                  // 多模态支持
"image_token_id": 248056
"video_token_id": 248057
"architectures": ["Qwen3_5MoeForConditionalGeneration"]
```

**Qwen3-Next 特有**:
```json
"decoder_sparse_step": 1
"norm_topk_prob": true
"output_router_logits": false
"use_sliding_window": false
"architectures": ["Qwen3NextForCausalLM"]
```

## 3. 架构认知修正

### 3.1 之前误解（已纠正）

❌ **错误认知**: Qwen3-Next 使用 Mamba (M-M-M-* pattern)

✅ **实际情况**: Qwen3-Next 使用 **Linear Attention**，与 Qwen3.5 架构类似！

### 3.2 实际架构对比

```
Qwen3-Next-80B-A3B:
- 48 layers (HF 配置)
- Layer pattern: 3x Linear Attention + 1x Full Attention (每4层循环)
- 512 experts, topk=10

Qwen3.5-35B-A3B:
- 40 layers  
- Layer pattern: 3x Linear Attention + 1x Full Attention (每4层循环)
- 256 experts, topk=8
```

### 3.3 PAI-Megatron-Patch 配置差异说明

**Qwen3-Next 在 PAI 脚本中** (`run_mcore_qwen3.sh`):
```bash
NUM_LAYERS=96  # 注意：不是 HF 的 48！
```

可能原因：
- PAI 可能使用 2x 计数（包含辅助层）
- `hybrid_override_pattern` 使用字符计数（24 组 × 4 字符 = 96）
- 或者是脚本版本差异

## 4. PAI-Megatron-Patch 支持评估

### 4.1 ✅ 已支持

| 特性 | 支持状态 | 说明 |
|------|---------|------|
| **Linear Attention** | ✅ 支持 | PAI 的 `qwen3_next` 实现 |
| **MoE (256/512专家)** | ✅ 支持 | 标准 Megatron MoE |
| **GQA (2头)** | ✅ 支持 | `--group-query-attention` |
| **SiLU激活** | ✅ 支持 | `--swiglu` 或类似 |
| **RMSNorm (1e-6)** | ✅ 支持 | `--norm-epsilon` |
| **RoPE (theta=1e7)** | ✅ 支持 | `--rotary-base` |
| **partial_rotary_factor=0.25** | ✅ 支持 | `--rotary-percent` |

### 4.2 ⚠️ 需验证

| 特性 | 风险等级 | 说明 |
|------|---------|------|
| **MTP** | 🟡 中 | `--mtp-num-layers 1` 需验证 |
| **Vocab 248320** | 🟢 低 | 仅需调整 `--padded-vocab-size` |
| **MRoPE** | 🟢 低 | 文本训练可能无需特殊处理 |
| **多模态** | 🟡 中 | 纯文本训练可忽略 |

## 5. 适配难度重新评估

| 维度 | 相似度 | 说明 |
|------|-------|------|
| 基础架构 | **95%** | 都使用 Linear Attention + Full Attention |
| MoE 配置 | **85%** | 专家数减半，topk 减少 |
| 词表/Embedding | **70%** | 词表扩大 63%，需特别注意 |
| 附加功能 | **80%** | Qwen3.5 增加 MTP |

**总体适配难度**: **🟡 中低风险** (之前错误评估为高风险)

## 6. 关键适配参数（更新）

基于 HF 配置到 PAI 配置的映射：

```bash
# Qwen3.5-35B-A3B 推测配置
NUM_LAYERS=80        # 推测: 40 * 2 (如果 PAI 使用 2x 计数)
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_KEY_VALUE_HEADS=2
INTERMEDIATE_SIZE=5120    # 待确认
MOE_INTERMEDIATE_SIZE=512
MAX_POSITION_EMBEDDINGS=262144
VOCAB_SIZE=248320         # 注意：比 Qwen3-Next 大很多
ROPE_THETA=10000000
RMS_NORM_EPS=1e-6

# MoE 参数
NUM_EXPERTS=256      # HF: 256
ROUTER_TOPK=8        # HF: 8
SHARED_EXPERT_INTERMEDIATE_SIZE=512

# 新增: MTP
MTP_NUM_LAYERS=1
```

## 7. 建议的适配路径

1. **复用 Qwen3-Next 的 PAI 实现**（95% 可复用）
2. **调整参数**: 
   - layers: 40 (HF) → 80 (PAI推测)
   - experts: 256
   - topk: 8
   - vocab: 248320
3. **验证 MTP**: `--mtp-num-layers 1`
4. **测试权重转换**: 确保 Linear Attention 层权重正确映射

## 8. 参考资源

- [Qwen3-Next HF Config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
- [Qwen3.5-35B-A3B HF Config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [PAI-Megatron-Patch qwen3_next](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen3_next)

---

**报告更新时间**: 2026-03-02  
**基于数据**: HuggingFace 官方配置  
**修正内容**: 纠正 "Mamba" 错误，确认两者都使用 Linear Attention
