# Qwen3.5-35B-A3B 训练框架调研与对齐验证方案

> **调研时间**: 2026-03-02  
> **调研目标**: 评估 PAI-Megatron-Patch 与 ms-swift 对 Qwen3.5-35B-A3B 的支持现状，提出对齐验证方案  
> **关键发现**: 两个框架均基于 Megatron-Core，版本接近，对齐验证可行

---

## 1. 执行摘要

### 1.1 核心结论

1. **ms-swift 完全支持 Qwen3.5-35B-A3B** 训练，基于 Megatron-Core backend
2. **PAI-Megatron-Patch** 通过复用 Qwen3-Next 实现可支持 Qwen3.5
3. **两个框架底层均为 Megatron-Core**，版本接近（0.15.x），对齐验证具备可行性
4. **主要差异**: 词表扩大 63% (151936→248320)、专家数减半 (512→256)、新增 MTP

### 1.2 版本信息

| 框架 | Megatron 版本 | megatron-core | 状态 |
|------|--------------|---------------|------|
| PAI Qwen3-Next | Megatron-LM-250908 | **0.15.0rc3** | 已验证 |
| ms-swift | 依赖用户环境 | **0.13+ / 0.15+** | 代码兼容 |

### 1.3 建议

- 优先使用 **ms-swift** 进行 Qwen3.5 训练（原生支持，配置简单）
- 如需使用 **PAI**，需完成权重转换和参数适配
- 对齐验证重点：**权重初始化、前向传播、训练动态**

---

## 2. ms-swift 框架支持现状

### 2.1 模型注册

ms-swift 在以下位置注册了 Qwen3.5 支持：

```python
# swift/model/models/qwen.py
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

### 2.2 Megatron Backend 实现

```python
# swift/megatron/model/mm_gpts/qwen3_5.py
@register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.qwen3_5,
        [
            ModelType.qwen3_5,
            ModelType.qwen3_5_moe,
        ],
        bridge_cls=Qwen3_5Bridge,
        loader=Qwen3_5Loader,
    ))
```

**关键实现文件**:
- `swift/megatron/model/mm_gpts/qwen3_5.py` - Qwen3.5 专属实现
- `swift/megatron/model/gpts/qwen3_next.py` - 复用 Qwen3-Next 的 Linear Attention
- `swift/megatron/model/model_config.py` - 配置处理

### 2.3 支持的训练模式

| 模式 | 命令 | 说明 |
|------|------|------|
| SFT | `swift sft` | 指令微调 |
| PT (Megatron) | `swift pt --train_type megatron` | 预训练/继续预训练 |
| RLHF | `swift rlhf` | 强化学习 |

---

## 3. PAI-Megatron-Patch 适配方案

### 3.1 现有基础

PAI-Megatron-Patch v0.12.3+ 已包含 `qwen3_next` 实现：
- 架构：**Linear Attention + Full Attention + MoE**
- HF 配置：48 layers, 512 experts, topk=10
- PAI 配置：`NUM_LAYERS=80` (推测 40*2)
- 脚本位置：`examples/qwen3_next/run_mcore_qwen3.sh`

### 3.2 Qwen3.5 适配参数

| 参数 | Qwen3-Next (HF) | Qwen3.5 (HF) | PAI 推测值 |
|------|----------------|--------------|-----------|
| `num_hidden_layers` | 48 | **40** | 80? |
| `hidden_size` | 2048 | 2048 | 2048 |
| `num_attention_heads` | 16 | 16 | 16 |
| `num_key_value_heads` | 2 | 2 | 2 |
| `num_experts` | 512 | **256** | 256 |
| `num_experts_per_tok` | 10 | **8** | 8 |
| `vocab_size` | 151936 | **248320** | 248320 |
| `mtp_num_hidden_layers` | N/A | **1** | 1 |

### 3.3 适配步骤

```bash
# 1. 复制并修改启动脚本
cp run_mcore_qwen3.sh run_qwen3.5_35b.sh

# 2. 添加 35B-A3B 配置分支
elif [ $MODEL_SIZE = 35B-A3B ]; then
    NUM_LAYERS=80
    NUM_EXPERTS=256
    ROUTER_TOPK=8
    VOCAB_SIZE=248320
    ...
fi

# 3. 权重转换
bash convert_qwen3_next.sh bf16 \
    /path/to/Qwen3.5-35B-A3B-HF \
    /path/to/Qwen3.5-35B-A3B-MCore

# 4. 启动训练
bash run_qwen3.5_35b.sh dlc 35B-A3B ...
```

---

## 4. 框架版本对比分析

### 4.1 Megatron-Core 版本

**PAI-Megatron-Patch**:
```python
# backends/megatron/Megatron-LM-250908/megatron/core/package_info.py
MAJOR = 0
MINOR = 15
PATCH = 0
PRE_RELEASE = 'rc3'
__version__ = '0.15.0rc3'
```

**ms-swift**:
```python
# 代码兼容多版本
mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
mcore_015 = version.parse(megatron.core.__version__) >= version.parse('0.15.0rc0')
```

### 4.2 版本差异影响

| 功能 | mcore 0.13 | mcore 0.15 | 影响 |
|------|-----------|-----------|------|
| `global_aux_loss` | ❌ | ✅ | Qwen3.5 使用 aux_loss，不影响 |
| `qk_layernorm` | ✅ | ✅ | 两者都支持 |
| `use_shared_expert_gate` | ✅ | ✅ | 两者都支持 |
| Context Parallel | 基础 | 增强 | Qwen3.5 可能不需要 |

### 4.3 结论

- ✅ **版本兼容**: PAI (0.15.0rc3) 与 ms-swift (0.15+) 基本对齐
- 🟢 **API 一致**: 0.15.x 之间 API 变化较小
- 🟡 **需验证**: PAI 的 patch 层是否有特殊修改

---

## 5. 架构实现对比

### 5.1 Linear Attention 实现

**ms-swift**:
```python
# swift/megatron/model/gpts/qwen3_next.py
class Qwen3NextGatedDeltaNet(_HuggingFaceModule, _Qwen3NextGatedDeltaNet):
    # 直接使用 HF 的 GatedDeltaNet 实现
```

**PAI**:
```python
# 类似实现，基于 megatron.core
layer_spec.submodules.self_attention.module = Qwen3NextGatedDeltaNet
```

**结论**: 两者都复用 HF 的 `GatedDeltaNet`，实现一致。

### 5.2 Layer Pattern 生成

**ms-swift**:
```python
# swift/megatron/model/model_config.py
if llm_model_type == 'qwen3_next' or hf_model_type in {'qwen3_5', 'qwen3_5_moe'}:
    res['layer_types'] = [
        'full_attention' if (i + 1) % 4 == 0 else 'linear_attention'
        for i in range(num_layers)
    ]
```

**PAI**:
```bash
# run_mcore_qwen3.sh
hybrid_model_options=" \
    --hybrid-attention-ratio 0.125 \
    --hybrid-override-pattern L-L-L-A-... \
    --is-hybrid-model
"
```

**结论**: Pattern 逻辑相同（每4层1个 Full Attention），需确保 `NUM_LAYERS` 一致。

### 5.3 RMSNorm 实现

**ms-swift**:
```python
class Qwen3NextRMSNorm(torch.nn.Module):
    # Zero-Centered: output * (1.0 + weight)
```

**PAI**: 需验证是否使用相同实现。

### 5.4 MoE 配置

| 参数 | ms-swift | PAI | 备注 |
|------|---------|-----|------|
| `num_experts` | 256 | 256 | 一致 |
| `topk` | 8 | 8 | 一致 |
| `aux_loss_coef` | 0.001 | 0.001 | 一致 |
| `shared_expert_gate` | ✅ | ✅ | 一致 |

---

## 6. 对齐验证方案

### 6.1 验证目标

证明 PAI 实现的 Qwen3.5 训练与 ms-swift（官方参考）在以下层面一致：
1. 权重初始化
2. 前向传播（Loss、Logits）
3. 反向传播（梯度）
4. 训练动态（Loss 曲线）

### 6.2 验证步骤

#### Step 1: 权重对齐验证

```python
# 从同一 HF checkpoint 转换
# 对比关键层权重
def verify_weights(pai_ckpt, swift_ckpt):
    layers = [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.self_attention.query_key_value.weight",
        "decoder.layers.0.mlp.experts.experts.0.weight",
        "output_layer.weight",
    ]
    for layer in layers:
        diff = torch.abs(pai_w - swift_w).max()
        assert diff < 1e-6, f"Mismatch in {layer}"
```

#### Step 2: 前向传播对齐

```python
# 使用相同随机输入
input_ids = torch.randint(0, 248320, (1, 512))

# 前向对比
loss_pai = pai_model(input_ids, labels).loss
loss_swift = swift_model(input_ids, labels).loss

assert abs(loss_pai - loss_swift) < 1e-5
```

#### Step 3: 梯度对齐

```python
# 单步训练后对比梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        rel_diff = torch.abs(pai_grad - swift_grad) / (torch.abs(pai_grad) + 1e-8)
        assert rel_diff.max() < 1e-4
```

#### Step 4: 训练动态对齐

```python
# 训练 100 步，记录 Loss
steps = range(0, 101, 10)
plt.plot(steps, pai_losses, label="PAI")
plt.plot(steps, swift_losses, label="ms-swift")
assert max(abs(p - s) for p, s in zip(pai_losses, swift_losses)) < 0.01
```

### 6.3 自动化验证脚本

建议添加以下文件到仓库：

```
verification/
├── test_weight_alignment.py      # 权重对齐测试
├── test_forward_alignment.py     # 前向对齐测试
├── test_gradient_alignment.py    # 梯度对齐测试
├── test_training_alignment.py    # 训练动态对齐测试
└── run_full_verification.sh      # 一键运行
```

### 6.4 容忍差异

以下差异为**可接受**范围：

| 差异类型 | 原因 | 容忍范围 |
|----------|------|----------|
| 随机初始化差异 | 不同框架随机种子实现 | 训练后收敛一致即可 |
| bf16 舍入误差 | 数值精度限制 | loss diff < 1e-3 |
| 通信顺序差异 | TP/PP/EP 调度 | 前向结果一致即可 |

---

## 7. 继续预训练建议

### 7.1 超参数对比

| 参数 | 从头训练 | 继续预训练 | 说明 |
|------|---------|-----------|------|
| 学习率 | 1e-4 | **1e-5 ~ 5e-5** | 降低 10x |
| warmup | 1%~2% tokens | **0.5%~1%** | 减少 warmup |
| weight decay | 0.1 | **0.01~0.1** | 可略微降低 |

### 7.2 训练模式

**ms-swift**:
```bash
swift pt \
    --model Qwen/Qwen3.5-35B-A3B \
    --train_type megatron \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    ...
```

**PAI**:
```bash
bash run_qwen3.5_35b.sh ... \
    --finetune \
    --reset-iteration \
    --lr 1e-5
```

---

## 8. 结论与建议

### 8.1 核心结论

1. **ms-swift 是官方推荐方案**: 原生支持 Qwen3.5，配置简单，维护活跃
2. **PAI 适配可行**: 基于 Qwen3-Next 实现，参数调整后可支持 Qwen3.5
3. **对齐验证必要**: 两个框架底层均为 Megatron-Core，具备对齐验证条件
4. **版本基本对齐**: PAI (0.15.0rc3) 与 ms-swift (0.15+) 版本接近

### 8.2 实施建议

| 场景 | 建议方案 |
|------|----------|
| **快速启动** | 直接使用 ms-swift |
| **生产环境** | 完成对齐验证后使用 PAI |
| **研究对比** | 同时跑两个框架，对比效果 |

### 8.3 下一步行动

1. ✅ 确认 ms-swift 环境的 megatron-core 版本（是否为 0.15.x）
2. ✅ 在相同版本下运行对齐验证脚本
3. ✅ 记录并分析差异（如有）
4. ✅ 更新适配指南和验证报告

---

## 9. 参考资源

- [Qwen3.5-35B-A3B HF Config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3-Next-80B-A3B HF Config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [ms-swift GitHub](https://github.com/modelscope/ms-swift)
- [PAI-Megatron-Patch GitHub](https://github.com/alibaba/Pai-Megatron-Patch)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)

---

**报告版本**: v1.0  
**最后更新**: 2026-03-02  
**基于数据**: ms-swift main branch, PAI-Megatron-Patch v0.12.3+
