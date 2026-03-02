# Qwen3.5-35B-A3B PAI-Megatron-Patch 适配指南（已修正）

> **修正说明**: 本指南已根据 HuggingFace 官方配置更新，基于 Qwen3-Next 和 Qwen3.5 都使用 Linear Attention 的事实。

## 1. 现有基础

PAI-Megatron-Patch v0.12.3 已包含 `qwen3_next` 目录，支持 **Qwen3-Next-80B-A3B** 模型：
- 架构：**Linear Attention + Full Attention + MoE** (已纠正：不是 Mamba)
- HF 配置：48 layers, 512 experts, topk=10
- PAI 配置：`NUM_LAYERS=96` (可能是 2x 计数)
- 配置脚本：`examples/qwen3_next/run_mcore_qwen3.sh`
- 权重转换：`toolkits/distributed_checkpoints_convertor/scripts/qwen3_next/`

**Qwen3.5-35B-A3B HF 配置**:
- 40 layers, 256 experts, topk=8
- Linear Attention 参数与 Qwen3-Next 相同
- 词表扩大至 248320 (+63%)
- 新增 MTP (Multi-Token Prediction)

## 2. 确认的关键参数（基于 HF 配置）

| 参数 | Qwen3-Next-80B (HF) | Qwen3.5-35B (HF) | PAI 推测值 |
|------|--------------------|------------------|-----------|
| `num_hidden_layers` | 48 | **40** | 80? (2x) |
| `hidden_size` | 2048 | **2048** | 2048 |
| `num_attention_heads` | 16 | **16** | 16 |
| `num_key_value_heads` | 2 | **2** | 2 |
| `intermediate_size` | 5120 | **?** | 5120 |
| `moe_intermediate_size` | 512 | **512** | 512 |
| `num_experts` | 512 | **256** | 256 |
| `num_experts_per_tok` | 10 | **8** | 8 |
| `vocab_size` | 151936 | **248320** | 248320 |
| `max_position_embeddings` | 262144 | **262144** | 262144 |
| `rope_theta` | 10000000 | **10000000** | 10000000 |
| `rms_norm_eps` | 1e-6 | **1e-6** | 1e-6 |
| `mtp_num_hidden_layers` | N/A | **1** | 1 |

## 3. 适配步骤（不改动原仓库）

### 步骤 1：创建自定义启动脚本

复制 `run_mcore_qwen3.sh` 为 `run_qwen3.5_35b.sh`，添加新配置：

```bash
elif [ $MODEL_SIZE = 35B-A3B ]; then
    # 基础参数 (基于 HF 配置，PAI 可能使用 2x 计数)
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    NUM_LAYERS=80        # 推测: 40 * 2 (需验证)
    INTERMEDIATE_SIZE=5120
    MOE_INTERMEDIATE_SIZE=512
    MAX_POSITION_EMBEDDINGS=262144
    NUM_KEY_VALUE_HEADS=2
    ROPE_THETA=10000000
    NUM_EXPERTS=256      # HF: 256
    ROUTER_TOPK=8        # HF: 8
    RMS_NORM_EPS=1e-6
    VOCAB_SIZE=248320    # HF: 248320 (注意! 比 Qwen3-Next 大 63%)

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

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    # Linear Attention 配置 (复用 Qwen3-Next)
    # 注意: pattern 长度需匹配 NUM_LAYERS
    hybrid_model_options=" \
                --hybrid-attention-ratio 0.125 \
                --hybrid-mlp-ratio 0.5 \
                --hybrid-override-pattern L-L-L-*-L-L-L-*-... \
                --is-hybrid-model
    "
    
    # 新增: MTP 支持
    mtp_options=" \
                --mtp-num-layers 1
    "
fi
```

### 步骤 2：权重转换

复用 `qwen3_next` 的转换工具，注意词表大小变化：

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

# 创建自定义转换脚本 convert_qwen3.5_35b.sh
MODEL_SIZE=35B-A3B
LOAD_DIR=/path/to/Qwen3.5-35B-A3B-HF
SAVE_DIR=/path/to/Qwen3.5-35B-A3B-MCore
MG2HF=false
USE_CUDA=true
PR=bf16

# 注意: vocab_size 从 151936 变为 248320
# 转换工具需要正确处理 embedding 层
```

### 步骤 3：训练启动

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3_next

bash run_qwen3.5_35b.sh \
    dlc \
    35B-A3B \
    1 \                          # micro_batch_size
    8 \                          # global_batch_size
    1e-5 \                       # lr
    1e-6 \                       # min_lr
    4096 \                       # seq_len
    4096 \                       # pad_len
    bf16 \                       # precision
    2 \                          # TP
    2 \                          # PP
    1 \                          # CP
    1 \                          # ETP
    4 \                          # EP (256 experts 可被 4/8 整除)
    true \                       # SP
    true \                       # DO
    false \                      # FL
    false \                      # SFT
    sel \                        # AC
    false \                      # OPTIMIZER_OFFLOAD
    10000 \                      # SAVE_INTERVAL
    /mnt/datasets/your_data \    # DATASET_PATH
    /mnt/datasets/your_data \    # VALID_DATASET_PATH
    /mnt/ckpts/Qwen3.5-35B-A3B-MCore \  # PRETRAIN_CHECKPOINT_PATH
    100000000 \                  # TRAIN_TOKENS
    10000 \                      # WARMUP_TOKENS
    /mnt/output                  # OUTPUT_BASEPATH
```

## 4. 关键注意事项（更新）

### 4.1 Linear Attention 架构
- 不是 Mamba！复用 Qwen3-Next 的 Linear Attention 实现
- `hybrid_override_pattern` 长度需与 `NUM_LAYERS` 匹配
- 每 4 层一个循环: 3x Linear Attention + 1x Full Attention

### 4.2 词表扩大 63%
```
Qwen3-Next: 151936
Qwen3.5:    248320
```
- 必须使用 `--padded-vocab-size 248320`
- 权重转换时需特别处理 embedding 和 lm_head 层

### 4.3 MTP (Multi-Token Prediction)
```json
"mtp_num_hidden_layers": 1
```
- PAI-Megatron-Patch 支持 `--mtp-num-layers`
- 需要验证与 MoE 的兼容性

### 4.4 MRoPE 说明
Qwen3.5 支持多模态 RoPE，但纯文本训练无需特殊处理：
- `--rotary-percent 0.25` (与 Qwen3-Next 相同)
- `--rotary-base 10000000` (相同)

### 4.5 内存优化建议
35B 模型在 8xH20 (80GB) 上的推荐配置：
```bash
TP=2          # Tensor Parallel
PP=2          # Pipeline Parallel  
EP=4          # Expert Parallel (256 experts / 4 = 64 per rank)
AC=sel        # 选择性重计算
DO=true       # Distributed Optimizer
```

## 5. 继续预训练特定建议

### 5.1 超参数推荐

| 参数 | 从头训练 | 继续预训练 | 说明 |
|------|---------|-----------|------|
| **学习率 (lr)** | `1e-4` ~ `3e-4` | `1e-5` ~ `5e-5` | 比从头训练低 10x |
| **最小学习率 (min_lr)** | `1e-6` | `1e-6` ~ `5e-6` | 通常保持或略高 |
| **warmup tokens** | 1% ~ 2% 总 token | 0.5% ~ 1% 总 token | 继续训练可减少 warmup |
| **weight decay** | 0.1 | 0.01 ~ 0.1 | 可略微降低 |
| **gradient clipping** | 1.0 | 1.0 | 保持不变 |

### 5.2 训练模式选择

```bash
# 继续预训练 (推荐)
--finetune \
--reset-iteration \
--train-tokens 100000000000 \

# 关键参数说明：
# --finetune: 加载预训练权重作为初始化
# --reset-iteration: 重置迭代计数器
# --train-tokens: 总训练 token 数
```

### 5.3 数据准备

1. **数据格式**: MMAP (`.bin` + `.idx`)，与 qwen3_next 兼容
2. **数据混合**: 建议保持与 base 模型相似的领域分布
3. **数据量**: 继续预训练建议至少 50B ~ 100B tokens

### 5.4 验证流程

```bash
# Step 1: 干跑验证 (不加载真实数据，验证配置)
bash run_qwen3.5_35b.sh ... --train-iters 10 --eval-iters 0

# Step 2: 单 batch 验证 (确认 loss 下降)
bash run_qwen3.5_35b.sh ... --train-iters 100 --micro-batch-size 1

# Step 3: 小规模验证 (1% 数据，确认整体流程)
# Step 4: 全量预训练
```

### 5.5 监控指标

| 指标 | 预期行为 | 异常信号 |
|------|---------|---------|
| **loss** | 平稳下降，初始约 2.0~2.5 | 突然飙升或 NaN |
| **learning rate** | 按 schedule 变化 | 不变化或跳跃 |
| **load balancing loss** | 0.001 左右 | 超过 0.01 |
| **gradient norm** | 1.0 左右 | 超过 10 或接近 0 |
| **throughput** | 稳定 | 突然下降 |

## 6. 验证清单（更新）

启动训练前确认：
- [x] 已确认使用 **Linear Attention** (不是 Mamba)
- [ ] `NUM_LAYERS` 设置正确 (HF: 40, PAI 可能: 80)
- [ ] `--padded-vocab-size 248320` 已设置
- [ ] 权重转换成功，特别注意 embedding 层
- [ ] `--mtp-num-layers 1` 已添加并验证
- [ ] `hybrid_override_pattern` 长度与层数匹配
- [ ] 各并行度参数可被整除（`num_layers % PP == 0`, `num_experts % EP == 0`）
- [ ] 数据集已转换为 MMAP 格式（`.bin` + `.idx`）
- [ ] **继续预训练**: `--finetune` 和 `--reset-iteration` 已设置
- [ ] **继续预训练**: 学习率已调整为 `1e-5` ~ `5e-5`
- [ ] **继续预训练**: warmup tokens 已设置为 0.5% ~ 1%

## 7. 快速启动模板

```bash
# 1. 下载模型
modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir ./Qwen3.5-35B-A3B

# 2. 检查 HF 配置
cat ./Qwen3.5-35B-A3B/config.json | grep -E "num_hidden_layers|num_experts|vocab_size"

# 3. 转换权重 (注意 vocab_size=248320)
bash convert_qwen3.5_35b.sh bf16

# 4. 验证转换结果
ls -lh /mnt/ckpts/Qwen3.5-35B-A3B-MCore/

# 5. 启动训练
bash run_qwen3.5_35b.sh dlc 35B-A3B 1 8 1e-5 1e-6 4096 4096 bf16 2 2 1 1 4 true true false false sel false 10000 ./data ./data ./ckpts 100000000 10000 ./output
```

## 8. 故障排查

### 8.1 词表大小不匹配
```
Error: embedding size mismatch
```
**解决**: 确认 `--padded-vocab-size 248320`

### 8.2 Layer pattern 长度错误
```
Error: hybrid_override_pattern length != num_layers
```
**解决**: 调整 pattern 长度匹配 `NUM_LAYERS`

### 8.3 Expert 数不能被 EP 整除
```
Error: num_experts must be divisible by EP
```
**解决**: 256 experts 可被 1, 2, 4, 8, 16 整除，选择合适的 EP

### 8.4 Loss 发散
```
Loss suddenly increases to NaN or Inf
```
**可能原因**:
- 学习率过高（继续预训练建议 `1e-5` ~ `5e-5`）
- warmup 不足
- 数据质量问题

**解决**:
- 降低学习率
- 增加 warmup tokens
- 检查数据预处理

### 8.5 Checkpoint 加载失败
```
Error loading checkpoint: shape mismatch
```
**解决**:
- 确认 `--finetune` 模式
- 检查 `NUM_LAYERS` 与 checkpoint 一致
- 确认 `VOCAB_SIZE` 与 checkpoint 一致

## 9. 替代方案：ms-swift

如果 PAI-Megatron-Patch 适配遇到困难，**ms-swift 已原生支持 Qwen3.5-MoE**：

```bash
# 使用 ms-swift 进行 Megatron 训练
swift sft \
    --model Qwen/Qwen3.5-35B-A3B \
    --train_type megatron \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --dataset your_dataset.jsonl \
    --output_dir output
```

**ms-swift 优势**:
- 原生支持 Qwen3.5-MoE (`qwen3_5_moe`)
- 自动处理 Linear Attention layer pattern
- 支持 MTP 训练
- 更简洁的配置

**参考**: [ms-swift 调研报告](./ms-swift-Qwen3-Next-vs-Qwen3.5-Research.md)

---

**总结**: Qwen3.5-35B-A3B 与 Qwen3-Next-80B-A3B 架构高度相似（都使用 Linear Attention），适配难度 **中低**。主要变化：层数减少、专家数减半、词表扩大、新增 MTP。

**继续预训练关键**: 
- 学习率降低 10x (`1e-5` ~ `5e-5`)
- 使用 `--finetune --reset-iteration` 模式
- 减少 warmup 比例

**最后更新**: 2026-03-02  
**基于配置**: HuggingFace Qwen3-Next 和 Qwen3.5 官方配置
