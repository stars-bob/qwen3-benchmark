# ms-swift 对 Qwen3-Next 和 Qwen3.5 的支持调研报告

> **调研时间**: 2026-03-02  
> **调研对象**: ms-swift 框架 v4.x  
> **目标模型**: Qwen3-Next-80B-A3B, Qwen3.5-35B-A3B

---

## 1. 执行摘要

**结论**: ms-swift **同时支持** Qwen3-Next 和 Qwen3.5 两个模型系列的训练。两者架构高度相似（都使用 Linear Attention + MoE），适配难度评估为 **中低**。

**核心发现**:
- ✅ 两者都使用 Linear Attention（不是 Mamba）
- ⚠️ 最大差异：词表扩大 63%（151936 → 248320）
- 🆕 Qwen3.5 新增 MTP 和 MRoPE 支持

---

## 2. ms-swift 支持矩阵

| 模型系列 | 模型类型常量 | 具体模型 | HuggingFace 架构类 | 支持状态 |
|----------|-------------|----------|-------------------|----------|
| **Qwen3-Next** | `qwen3_next` | Qwen3-Next-80B-A3B-Instruct<br>Qwen3-Next-80B-A3B-Thinking<br>Qwen3-Coder-Next | `Qwen3NextForCausalLM` | ✅ 完全支持 |
| **Qwen3.5-MoE** | `qwen3_5_moe` | Qwen3.5-35B-A3B<br>Qwen3.5-122B-A10B<br>Qwen3.5-397B-A17B | `Qwen3_5MoeForConditionalGeneration` | ✅ 完全支持 |
| **Qwen3.5-Dense** | `qwen3_5` | Qwen3.5-27B | `Qwen3_5ForConditionalGeneration` | ✅ 完全支持 |

**代码位置**:
- 模型注册: `swift/model/models/qwen.py`
- Megatron 支持: `swift/megatron/model/gpts/qwen3_next.py`
- Template: `swift/template/templates/qwen.py`

---

## 3. 架构差异详细对比

### 3.1 基础架构参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| **架构类型** | Linear Attention + MoE | Linear Attention + MoE | ✅ 相同 |
| `num_hidden_layers` | 48 | **40** | -8 (-17%) |
| `hidden_size` | 2048 | **2048** | = |
| `num_attention_heads` | 16 | **16** | = |
| `num_key_value_heads` | 2 | **2** | = |
| `head_dim` | 256 | **256** | = |
| `moe_intermediate_size` | 512 | **512** | = |
| `hidden_act` | "silu" | **"silu"** | = |
| `vocab_size` | 151936 | **248320** | **+63%** ⚠️ |
| `max_position_embeddings` | 262144 | **262144** | = |

### 3.2 MoE 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| `num_experts` | 512 | **256** | -50% |
| `num_experts_per_tok` (topk) | 10 | **8** | -2 |
| `shared_expert_intermediate_size` | 512 | **512** | = |
| `router_aux_loss_coef` | 0.001 | **0.001** | = |

### 3.3 Linear Attention 参数

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B | 变化 |
|------|-------------------|-----------------|------|
| `linear_conv_kernel_dim` | 4 | **4** | = |
| `linear_key_head_dim` | 128 | **128** | = |
| `linear_num_key_heads` | 16 | **16** | = |
| `linear_num_value_heads` | 32 | **32** | = |
| `linear_value_head_dim` | 128 | **128** | = |
| `full_attention_interval` | 4 | **4** | = |

**结论**: Linear Attention 实现完全一致！

### 3.4 Qwen3.5 新增特性

| 特性 | 说明 | 对训练的影响 |
|------|------|-------------|
| **MTP** | `mtp_num_hidden_layers: 1` | 支持多 token 预测，纯文本训练可选启用 |
| **MRoPE** | `mrope_section: [11, 11, 10]` | 多模态 RoPE，纯文本训练可忽略 |
| **多模态** | `vision_config` | 文本训练可忽略 |
| **新词表 token** | `image_token_id: 248056`<br>`video_token_id: 248057` | 需确保不干扰文本训练 |

### 3.5 已移除/变更参数

| 参数 | Qwen3-Next | Qwen3.5 | 说明 |
|------|-----------|---------|------|
| `decoder_sparse_step` | ✅ 有 | ❌ 无 | Qwen3.5 移除 |
| `norm_topk_prob` | ✅ 有 | ❌ 无 | Qwen3.5 移除 |
| `architectures` | `Qwen3NextForCausalLM` | `Qwen3_5MoeForConditionalGeneration` | 架构类名变更 |

---

## 4. 风险点验证

### 4.1 之前列的风险点（已验证）

| 风险点 | 原评估 | 实际状态 | 说明 |
|--------|--------|----------|------|
| **Mamba 架构** | 🔴 高风险 | ❌ **错误** | 两者都用 Linear Attention，不是 Mamba |
| **词表扩大 63%** | 🟡 中风险 | ⚠️ **确认** | 151936 → 248320，embedding 层需特别注意 |
| **Linear Attention 兼容性** | 🟡 中风险 | ✅ **兼容** | 两者使用相同的 `GatedDeltaNet` 实现 |
| **层数差异** | 🟡 中风险 | ⚠️ **确认** | 40 vs 48，但 ms-swift/PAI 可能使用 2x 计数 |

### 4.2 新发现的风险点

| 风险点 | 等级 | 说明 | 建议 |
|--------|------|------|------|
| **密集 vs MoE 分离** | 🟡 中 | `qwen3_5` 和 `qwen3_5_moe` 是独立的模型类型 | 确保使用正确的 model_type |
| **MTP + MoE 兼容性** | 🟡 中 | Qwen3.5 的 MTP 层与 MoE 的交互需验证 | 小规模测试后再大规模训练 |
| **MRoPE 文本训练** | 🟢 低 | 纯文本训练可忽略 MRoPE | 确保 `--rotary-percent 0.25` 正确设置 |
| **HF 与 PAI 层数映射** | 🟡 中 | HF 40 层 → PAI 可能 80 层 | 需测试验证实际层数 |
| **Shared Expert Gate** | 🟢 低 | 两者都启用 `use_shared_expert_gate` | 行为一致，无风险 |
| **新词表 token 干扰** | 🟡 中 | image/video token id 在文本词表范围内 | 确保文本数据不包含这些特殊 token |

---

## 5. ms-swift Megatron 实现分析

### 5.1 Layer 类型生成逻辑

```python
# swift/megatron/model/model_config.py
if llm_model_type == 'qwen3_next' or hf_model_type in {'qwen3_5', 'qwen3_5_moe'}:
    full_attention_interval = res.pop('full_attention_interval', 4)
    num_layers = res['num_layers']
    res['layer_types'] = [
        'full_attention' if (i + 1) % full_attention_interval == 0 else 'linear_attention'
        for i in range(num_layers)
    ]
```

**结论**: Qwen3-Next 和 Qwen3.5 使用完全相同的 layer pattern 生成逻辑。

### 5.2 关键组件实现

| 组件 | 实现类 | 说明 |
|------|--------|------|
| **Linear Attention** | `Qwen3NextGatedDeltaNet` | 继承自 HF 的 `Qwen3NextGatedDeltaNet` |
| **Full Attention** | `Qwen3NextSelfAttention` | 支持 gated 机制 (`torch.sigmoid(gate)`) |
| **RMSNorm** | `Qwen3NextRMSNorm` | Zero-Centered: `(1 + weight)` 而非 `weight` |
| **MoE** | Megatron 标准 MoE | 支持 `use_shared_expert_gate` |

### 5.3 Qwen3.5 特殊处理

```python
# swift/megatron/model/gpts/qwen3_next.py
# qwen3.5 dense 特殊处理
if config.hf_model_type == 'qwen3_5':
    layer_spec.submodules.mlp.submodules.linear_fc1 = TEColumnParallelLinear
```

**注意**: Qwen3.5-Dense (27B) 在 MLP 层有特殊处理，但 MoE 版本与 Qwen3-Next 一致。

---

## 6. 适配建议

### 6.1 PAI-Megatron-Patch 适配要点

| 参数 | Qwen3-Next-80B | Qwen3.5-35B-A3B | 备注 |
|------|---------------|-----------------|------|
| `NUM_LAYERS` | 96 (48*2?) | **80** (推测 40*2) | 需验证 |
| `NUM_EXPERTS` | 512 | **256** | 可被 4/8 整除 |
| `ROUTER_TOPK` | 10 | **8** | - |
| `VOCAB_SIZE` | 151936 | **248320** | 必须准确设置 |
| `MTP_NUM_LAYERS` | 0 | **1** | Qwen3.5 新增 |

### 6.2 权重转换注意事项

1. **Embedding 层**: 词表从 151936 扩大到 248320，需确保转换工具正确处理
2. **MTP 层**: Qwen3.5 的 MTP 权重需要特别处理
3. **RMSNorm**: 使用 Zero-Centered 实现，无需 +1/-1 转换

### 6.3 训练配置建议

```bash
# Qwen3.5-35B-A3B 推荐配置 (8x H20 80GB)
TP=2          # Tensor Parallel
PP=2          # Pipeline Parallel
EP=4          # Expert Parallel (256 experts / 4 = 64 per rank)
AC=sel        # 选择性激活重计算
DO=true       # Distributed Optimizer
VOCAB_SIZE=248320  # 注意新词表大小
MTP_LAYERS=1  # 启用 MTP
```

---

## 7. 验证清单

在启动训练前，请确认以下事项：

- [ ] 已确认使用 **Linear Attention**（不是 Mamba）
- [ ] `NUM_LAYERS` 设置正确（HF: 40，PAI 可能: 80）
- [ ] `--padded-vocab-size 248320` 已正确设置
- [ ] 权重转换成功，特别注意 embedding 和 lm_head 层
- [ ] `--mtp-num-layers 1` 已添加（Qwen3.5 可选）
- [ ] `hybrid_override_pattern` 长度与层数匹配
- [ ] 各并行度参数可被整除：
  - `num_layers % PP == 0`
  - `num_experts % EP == 0` (256 可被 1/2/4/8/16 整除)
- [ ] 数据集已转换为 MMAP 格式（`.bin` + `.idx`）

---

## 8. 参考资源

- [Qwen3-Next HF Config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
- [Qwen3.5-35B-A3B HF Config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)
- [ms-swift GitHub](https://github.com/modelscope/ms-swift)
- [PAI-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)

---

## 9. 附录：关键代码引用

### 9.1 ms-swift 模型注册

```python
# swift/model/models/qwen.py

# Qwen3-Next 注册
register_model(ModelMeta(
    LLMModelType.qwen3_next,
    [
        ModelGroup([
            Model('Qwen/Qwen3-Next-80B-A3B-Instruct'),
            Model('Qwen/Qwen3-Next-80B-A3B-Instruct-FP8'),
        ], TemplateType.qwen3_nothinking),
    ],
    requires=['transformers>=4.57'],
    architectures=['Qwen3NextForCausalLM'],
))

# Qwen3.5-MoE 注册
register_model(ModelMeta(
    MLLMModelType.qwen3_5_moe,
    [
        ModelGroup([
            Model('Qwen/Qwen3.5-35B-A3B-Base', 'Qwen/Qwen3.5-35B-A3B-Base'),
            Model('Qwen/Qwen3.5-35B-A3B', 'Qwen/Qwen3.5-35B-A3B'),
            Model('Qwen/Qwen3.5-122B-A10B', 'Qwen/Qwen3.5-122B-A10B'),
            Model('Qwen/Qwen3.5-397B-A17B', 'Qwen/Qwen3.5-397B-A17B'),
        ], TemplateType.qwen3_5),
    ],
    architectures=['Qwen3_5MoeForConditionalGeneration'],
))
```

### 9.2 Megatron Layer Spec 生成

```python
# swift/megatron/model/gpts/qwen3_next.py

class Qwen3NextLoader(MegatronModelLoader):
    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        # ...
        for layer_type in self.config.layer_types:
            layer_spec = deepcopy(moe_layer_spec)
            if layer_type == 'linear_attention':
                layer_spec.submodules.self_attention.module = self.gated_delta_net
            elif layer_type == 'full_attention':
                layer_spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
                layer_spec.submodules.self_attention.module = Qwen3NextSelfAttention
            # Replace ALL layernorms with Qwen3NextRMSNorm (Zero-Centered)
            layer_spec.submodules.input_layernorm = layer_norm_impl
            # ...
```

---

**报告版本**: v1.0  
**最后更新**: 2026-03-02  
**基于 ms-swift 版本**: 4.x (main branch)
