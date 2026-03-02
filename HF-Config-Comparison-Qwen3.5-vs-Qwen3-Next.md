# Qwen3.5-35B-A3B vs Qwen3-Next-80B-A3B HF 配置对比报告

## 1. 配置来源

| 模型 | HF 地址 |
|------|---------|
| Qwen3-Next-80B-A3B | https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct |
| Qwen3.5-35B-A3B | https://huggingface.co/Qwen/Qwen3.5-35B-A3B |

## 2. HF 配置直接对比

### 2.1 基础架构参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **num_hidden_layers** | 48 | **40** | -8 (-17%) |
| **hidden_size** | 2048 | **2048** | = |
| **num_attention_heads** | 16 | **16** | = |
| **num_key_value_heads** | 2 | **2** | = |
| **head_dim** | 256 | **256** | = |
| **intermediate_size** | 5120 | **?** (未直接显示) | 待确认 |
| **moe_intermediate_size** | 512 | **512** | = |
| **hidden_act** | "silu" | **"silu"** | = |
| **vocab_size** | 151936 | **248320** | **+63%** |
| **max_position_embeddings** | 262144 | **262144** | = |

### 2.2 MoE 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **num_experts** | 512 | **256** | -256 (-50%) |
| **num_experts_per_tok** | 10 | **8** | -2 |
| **shared_expert_intermediate_size** | 512 | **512** | = |
| **router_aux_loss_coef** | 0.001 | **0.001** | = |
| **norm_topk_prob** | true | **未显示** | 可能相同 |

### 2.3 Linear Attention 参数 (关键发现!)

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **linear_conv_kernel_dim** | 4 | **4** | = |
| **linear_key_head_dim** | 128 | **128** | = |
| **linear_num_key_heads** | 16 | **16** | = |
| **linear_num_value_heads** | 32 | **32** | = |
| **linear_value_head_dim** | 128 | **128** | = |
| **full_attention_interval** | 4 | **4** | = |

**重要发现**: Qwen3-Next 也使用 Linear Attention，不是 Mamba！

### 2.4 RoPE 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **rope_theta** | 10000000 | **10000000** | = |
| **partial_rotary_factor** | 0.25 | **0.25** | = |
| **rope_scaling** | null | **(复杂对象)** | 差异 |

Qwen3.5 引入 MRoPE：
```json
"rope_parameters": {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "rope_type": "default",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25
}
```

### 2.5 归一化参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **rms_norm_eps** | 1e-06 | **1e-06** | = |

### 2.6 新增/移除参数

**Qwen3.5 新增**:
```json
"mtp_num_hidden_layers": 1
"mtp_use_dedicated_embeddings": false
"vision_config": {...}  // 多模态支持
"image_token_id": 248056
"video_token_id": 248057
```

**Qwen3-Next 特有**:
```json
"decoder_sparse_step": 1
"norm_topk_prob": true
"output_router_logits": false
"use_sliding_window": false
```

## 3. 架构认知修正

### 3.1 之前误解

❌ **错误**: Qwen3-Next 使用 Mamba (M-M-M-* pattern)

✅ **正确**: Qwen3-Next 也使用 Linear Attention，与 Qwen3.5 架构类似！

### 3.2 实际架构对比

```
Qwen3-Next-80B-A3B HF:
- 48 layers
- Layer pattern: 3x Linear Attention + 1x Full Attention (每4层循环)
- 512 experts, topk=10

Qwen3.5-35B-A3B HF:
- 40 layers  
- Layer pattern: 3x Linear Attention + 1x Full Attention (每4层循环)
- 256 experts, topk=8
```

### 3.3 PAI-Megatron-Patch 配置 vs HF 配置差异

**Qwen3-Next 在 PAI 中** (`run_mcore_qwen3.sh`):
```bash
NUM_LAYERS=96  # 不是 48!
NUM_EXPERTS=512
ROUTER_TOPK=10
hybrid_override_pattern="M-M-M-*-..."
```

**说明**: PAI 脚本中的 `NUM_LAYERS=96` 可能是：
1. 包含 MTP 层的总层数
2. 或者是脚本错误/旧版本配置
3. 或者 `hybrid_override_pattern` 包含 96 个字符（24 组 × 4）

## 4. PAI-Megatron-Patch 适配推测

### 4.1 假设 PAI 对 Qwen3-Next 的支持是准确的

基于 HF 配置到 PAI 配置的映射关系：

| HF 参数 | PAI 参数 | Qwen3-Next 值 | 推测 Qwen3.5 值 |
|---------|---------|---------------|----------------|
| num_hidden_layers | NUM_LAYERS | 48→96 | 40→**80?** |
| num_experts | NUM_EXPERTS | 512 | **256** |
| num_experts_per_tok | ROUTER_TOPK | 10 | **8** |
| hidden_size | HIDDEN_SIZE | 2048 | **2048** |
| intermediate_size | INTERMEDIATE_SIZE | 5120 | **?** |
| moe_intermediate_size | MOE_INTERMEDIATE_SIZE | 512 | **512** |
| vocab_size | VOCAB_SIZE | 151936 | **248320** |
| max_position_embeddings | MAX_POSITION_EMBEDDINGS | 262144 | **262144** |

### 4.2 Qwen3.5 特有挑战

#### (1) 词表扩大 63%
```
Qwen3-Next: 151936
Qwen3.5:    248320
```
需要确认 `--padded-vocab-size` 和 tokenizer 兼容性。

#### (2) MTP (Multi-Token Prediction)
```json
"mtp_num_hidden_layers": 1
```
PAI-Megatron-Patch 支持 MTP，但需要验证：
- `--mtp-num-layers 1` 参数
- 与 MoE 的兼容性

#### (3) MRoPE (Multi-modal RoPE)
```json
"mrope_interleaved": true
"mrope_section": [11, 11, 10]
```
文本-only 训练可能无需特殊处理，但需要注意 `--rotary-percent` 设置。

#### (4) 多模态支持
```json
"vision_config": {...}
```
如果是纯文本训练，可以忽略；如果需要 VL 训练，需要额外支持。

## 5. 更新的适配建议

### 5.1 最小改动启动脚本

```bash
elif [ $MODEL_SIZE = 35B-A3B ]; then
    # 基础参数 (基于 HF 配置映射)
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    NUM_LAYERS=80        # 推测: 40 * 2 (如果 PAI 使用 2x 计数)
    INTERMEDIATE_SIZE=5120   # 待确认
    MOE_INTERMEDIATE_SIZE=512
    MAX_POSITION_EMBEDDINGS=262144
    NUM_KEY_VALUE_HEADS=2
    ROPE_THETA=10000000
    NUM_EXPERTS=256      # HF: 256
    ROUTER_TOPK=8        # HF: 8
    RMS_NORM_EPS=1e-6
    VOCAB_SIZE=248320    # HF: 248320 (+63%)
    
    # MoE 选项
    moe_options=" \
        --moe-grouped-gemm \
        --moe-token-dispatcher-type alltoall \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff 0.001 \
        --moe-shared-expert-intermediate-size 512 \
        "
    
    # 新增: MTP 支持
    mtp_options=" \
        --mtp-num-layers 1 \
        "
    
    # Hybrid 模式 (复用 Qwen3-Next 的 pattern)
    # 注意: 可能需要调整 pattern 长度匹配 40 层
    hybrid_model_options=" \
        --hybrid-attention-ratio 0.125 \
        --hybrid-mlp-ratio 0.5 \
        --hybrid-override-pattern ... \
        --is-hybrid-model \
        --linear-conv-kernel-dim 4 \
        --linear-key-head-dim 128 \
        --linear-value-head-dim 128
    "
fi
```

### 5.2 关键待确认项

| 项 | 风险 | 验证方法 |
|----|------|---------|
| NUM_LAYERS=80? | 🟡 中 | 查看 PAI 内部如何计数 |
| intermediate_size | 🟡 中 | HF 未直接显示，需推断 |
| Vocab 248320 | 🟢 低 | 仅需调整参数 |
| MTP 兼容性 | 🟡 中 | 测试 `--mtp-num-layers` |
| Linear Attention | 🟢 低 | 复用 Qwen3-Next 实现 |

## 6. 结论

### 6.1 架构相似度评估

| 维度 | 相似度 | 说明 |
|------|-------|------|
| 基础架构 | **90%** | 都使用 Linear Attention + Full Attention |
| MoE 配置 | **85%** | 专家数减半，topk 减少 |
| 词表/Embedding | **70%** | 词表扩大 63%，需特别注意 |
| 附加功能 | **60%** | Qwen3.5 增加 MTP、多模态 |

### 6.2 适配难度重新评估

| 模型 | 之前评估 | 修正后 |
|------|---------|--------|
| Qwen3-Next → PAI | ✅ 已支持 | ✅ 已支持 |
| Qwen3.5 → PAI | 🔴 高风险 | **🟡 中低风险** |

**主要变化**:
- ✅ 两者都使用 Linear Attention（不是 Mamba）
- ⚠️ 词表扩大需要调整 embedding 层
- ⚠️ MTP 需要验证支持
- 🟢 其他架构参数高度相似

### 6.3 建议的适配路径

1. **复用 Qwen3-Next 的 PAI 实现**（90% 可复用）
2. **调整参数**: layers (40), experts (256), topk (8), vocab (248320)
3. **验证 MTP**: `--mtp-num-layers 1`
4. **测试权重转换**: 确保 Linear Attention 层权重正确映射

---

**报告更新时间**: 2026-03-02  
**数据来源**: HuggingFace Qwen3-Next 和 Qwen3.5 官方配置
