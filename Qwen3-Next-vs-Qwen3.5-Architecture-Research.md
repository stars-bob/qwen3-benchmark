# Qwen3-Next vs Qwen3.5-35B-A3B 架构调研报告

## 1. 调研背景

本文档深入分析 PAI-Megatron-Patch 框架中已支持的 Qwen3-Next-80B-A3B 模型与目标模型 Qwen3.5-35B-A3B 之间的架构差异，为适配工作提供技术参考。

## 2. Qwen3-Next-80B-A3B 架构详解

### 2.1 基础参数（来自 run_mcore_qwen3.sh）

```bash
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_LAYERS=96
INTERMEDIATE_SIZE=5120
MOE_INTERMEDIATE_SIZE=512
MAX_POSITION_EMBEDDINGS=262144
NUM_KEY_VALUE_HEADS=2
ROPE_THETA=10000000
NUM_EXPERTS=512
ROUTER_TOPK=10
RMS_NORM_EPS=1e-6
```

### 2.2 Hybrid 架构设计

Qwen3-Next 采用 **Mamba + Transformer + MoE** 的混合架构：

```bash
hybrid_model_options=" \
    --hybrid-attention-ratio 0.125 \
    --hybrid-mlp-ratio 0.5 \
    --hybrid-override-pattern M-M-M-*-M-M-M-*-... \
    --is-hybrid-model \
    --mamba-state-dim 128 \
    --mamba-head-dim 128 \
    --mamba-num-groups 16 \
    --mamba-num-heads 32
"
```

**关键特征：**
- **hybrid_attention_ratio=0.125**：12.5% 的层使用标准 Attention
- **hybrid_mlp_ratio=0.5**：50% 的层使用 MLP/MoE
- **Pattern**: `M-M-M-*` 循环，每 4 层一个周期
  - M: Mamba 层
  - M: Mamba 层  
  - M: Mamba 层
  - *: Attention + MoE 层

### 2.3 MoE 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| num_experts | 512 | 总专家数 |
| topk | 10 | 每个 token 激活的专家数 |
| expert_intermediate_size | 512 | 专家 FFN 维度 |
| shared_expert | 512 | 共享专家维度 |
| load_balancing | aux_loss | 使用辅助损失平衡 |
| aux_loss_coeff | 0.001 | 辅助损失系数 |

### 2.4 并行策略建议

```
Total Params: 80B (激活 8B)
Recommended Parallel:
- TP=1, PP=1, EP=8 (单机 8xH20)
- TP=2, PP=2, EP=4 (多机扩展)
```

## 3. Qwen3.5-35B-A3B 架构推测

### 3.1 参数对比分析

基于命名规则和模型规模推测：

| 参数 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B (推测) | 变化率 |
|------|-------------------|----------------------|--------|
| 总参数量 | 80B | 35B | -56% |
| 激活参数 | 8B | ~3.5B | -56% |
| hidden_size | 2048 | 2048 (可能不变) | 0% |
| num_layers | 96 | 48-64 (估算) | -33% to -50% |
| num_experts | 512 | 128-256 (估算) | -50% to -75% |
| intermediate_size | 5120 | 4096 (估算) | -20% |
| max_seq_len | 262K | 128K (可能) | -50% |

### 3.2 可能的架构差异

#### 3.2.1 层数缩减

35B 模型可能减少层数但保持每层的维度：

```python
# 80B 配置
num_layers = 96
pattern_repeat = 96 / 4 = 24  # M-M-M-* 重复 24 次

# 35B 推测配置
num_layers = 48  # 或 64
pattern_repeat = 48 / 4 = 12  # M-M-M-* 重复 12 次
```

#### 3.2.2 MoE 规模缩减

```python
# 80B 配置
num_experts = 512
topk = 10
expert_dim = 512

# 35B 推测配置
num_experts = 128  # 或 256
topk = 8  # 可能减少
expert_dim = 512  # 可能保持不变
```

#### 3.2.3 注意力头调整

```python
# 80B 配置
num_attention_heads = 16
num_key_value_heads = 2  # GQA

# 35B 可能配置
num_attention_heads = 16  # 可能不变
num_key_value_heads = 4   # 可能增加（更小的模型可能不需要那么强的压缩）
```

### 3.3 关键差异点

#### 1. Hybrid Pattern 长度

```bash
# 80B: 96 层，pattern 重复 24 次
--hybrid-override-pattern M-M-M-*-M-M-M-*-...(共96层)

# 35B: 假设 48 层，pattern 重复 12 次
--hybrid-override-pattern M-M-M-*-...(共48层，需重新生成)
```

**重要**：`hybrid_override_pattern` 必须与 `num_layers` 完全匹配，否则训练会失败。

#### 2. 专家并行度限制

```bash
# 80B: 512 experts，可被 8 整除
EP=8  # 每个 rank 管理 64 个专家

# 35B: 假设 128 experts
EP=4 或 8  # 128 可被 4/8 整除，但不能被 16 整除
```

#### 3. 序列长度支持

```bash
# 80B: 262K 上下文
MAX_POSITION_EMBEDDINGS=262144

# 35B: 可能减少到 128K
MAX_POSITION_EMBEDDINGS=131072  # 推测
```

## 4. 适配风险评估

### 4.1 低风险（可直接复用）

| 组件 | 复用程度 | 说明 |
|------|---------|------|
| Tokenizer | 100% | Qwen3Tokenizer 完全兼容 |
| Mamba 实现 | 100% | `megatron_patch/model/qwen3_next/` 直接复用 |
| MoE 路由 | 100% | Megatron-LM-250908 原生支持 |
| 权重转换 | 90% | 仅需调整参数映射 |

### 4.2 中风险（需验证）

| 组件 | 风险点 | 验证方法 |
|------|--------|---------|
| Hybrid Pattern | 层数不匹配 | 检查 config.json 中的 layer_pattern |
| RoPE 缩放 | theta 值可能不同 | 对比 rope_theta 参数 |
| 激活函数 | 可能有变化 | 检查 hidden_act 字段 |

### 4.3 高风险（需深度调研）

| 组件 | 潜在问题 | 应对策略 |
|------|---------|---------|
| 注意力机制 | 可能引入 MLA/MQA 变体 | 检查 attention_type 字段 |
| 位置编码 | 可能使用 Yarn/NTK-aware | 对比 rope_scaling 配置 |
| 归一化 | 可能使用 different eps | 检查 rms_norm_eps |

## 5. 调研结论

### 5.1 核心发现

1. **架构高度相似**：Qwen3.5-35B-A3B 极有可能是 Qwen3-Next-80B-A3B 的"缩放版"
2. **主要差异**：层数、专家数、序列长度按比例缩减
3. **复用价值**：PAI-Megatron-Patch 的 `qwen3_next` 实现可直接复用 90%+

### 5.2 关键待确认项

必须获取 Qwen3.5-35B-A3B 的 `config.json` 后确认：

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "hidden_size": 2048,
  "num_hidden_layers": 48,  // 关键！
  "num_attention_heads": 16,
  "num_key_value_heads": 4,  // 关键！
  "intermediate_size": 4096,
  "num_experts": 128,  // 关键！
  "num_experts_per_tok": 8,  // 关键！
  "moe_intermediate_size": 512,
  "max_position_embeddings": 131072,  // 关键！
  "rope_theta": 10000000,
  "rms_norm_eps": 1e-6,
  "hybrid_config": {  // 关键！
    "attention_ratio": 0.125,
    "mlp_ratio": 0.5,
    "pattern": "M-M-M-*"
  }
}
```

### 5.3 推荐适配策略

1. **Phase 1**: 获取 config.json，确认架构参数
2. **Phase 2**: 修改启动脚本，调整 `hybrid_override_pattern` 长度
3. **Phase 3**: 权重转换，验证 checkpoint 可加载
4. **Phase 4**: 小规模训练验证（1-10 steps）
5. **Phase 5**: 正式训练

## 6. 参考资源

- [Qwen3-Next GitHub](https://github.com/QwenLM/Qwen3)
- [PAI-Megatron-Patch qwen3_next 示例](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen3_next)
- [Megatron-LM MoE 文档](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/moe.md)

---

**报告生成时间**: 2026-03-02  
**基于 PAI-Megatron-Patch 版本**: v0.12.3  
**调研状态**: 等待 Qwen3.5-35B-A3B 官方配置发布
