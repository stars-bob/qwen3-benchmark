# Qwen3.5-35B-A3B PAI-Megatron-Patch 适配指南

> 本文档提供在不改动原仓库的前提下，将 Qwen3.5-35B-A3B 模型适配到 PAI-Megatron-Patch 框架的训练建议。

## 1. 现有基础

PAI-Megatron-Patch v0.12.3 已包含 `qwen3_next` 目录，支持 **Qwen3-Next-80B-A3B** 模型：
- 架构：Hybrid Mamba-Transformer-MoE
- 配置脚本：`examples/qwen3_next/run_mcore_qwen3.sh`
- 权重转换：`toolkits/distributed_checkpoints_convertor/scripts/qwen3_next/`

Qwen3.5-35B-A3B 预计是类似架构但参数更小的变体，可直接复用现有基础设施。

## 2. 需要确认的关键参数

建议从 HuggingFace 下载 `Qwen3.5-35B-A3B` 的 `config.json`，获取以下参数：

| 参数 | Qwen3-Next-80B-A3B (参考) | Qwen3.5-35B-A3B (需确认) |
|------|--------------------------|-------------------------|
| `hidden_size` | 2048 | ? |
| `num_hidden_layers` | 96 | ? |
| `num_attention_heads` | 16 | ? |
| `num_key_value_heads` | 2 | ? |
| `intermediate_size` | 5120 | ? |
| `moe_intermediate_size` | 512 | ? |
| `num_experts` | 512 | ? |
| `num_experts_per_tok` (topk) | 10 | ? |
| `max_position_embeddings` | 262144 | ? |
| `rope_theta` | 10000000 | ? |
| `rms_norm_eps` | 1e-6 | ? |
| `hybrid_attention_ratio` | 0.125 | ? |
| `hybrid_mlp_ratio` | 0.5 | ? |
| `hybrid_override_pattern` | M-M-M-*-... | ? |

## 3. 适配步骤（不改动原仓库）

### 步骤 1：创建自定义启动脚本

复制 `run_mcore_qwen3.sh` 为 `run_qwen3.5_35b.sh`，在 `elif [ $MODEL_SIZE = A3B ]` 前添加新配置：

```bash
elif [ $MODEL_SIZE = 35B-A3B ]; then
    HIDDEN_SIZE=2048           # 根据实际 config 修改
    NUM_ATTENTION_HEADS=16     # 根据实际 config 修改
    NUM_LAYERS=72              # 根据实际 config 修改（估计值）
    INTERMEDIATE_SIZE=4096     # 根据实际 config 修改
    MOE_INTERMEDIATE_SIZE=512  # 根据实际 config 修改
    MAX_POSITION_EMBEDDINGS=131072  # 根据实际 config 修改
    NUM_KEY_VALUE_HEADS=4      # 根据实际 config 修改
    ROPE_THETA=10000000
    NUM_EXPERTS=128            # 根据实际 config 修改（估计值）
    ROUTER_TOPK=8              # 根据实际 config 修改
    RMS_NORM_EPS=1e-6

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
    
    # 根据 config 中的 hybrid 配置调整
    hybrid_model_options=" \
                --hybrid-attention-ratio 0.125 \
                --hybrid-mlp-ratio 0.5 \
                --hybrid-override-pattern M-M-M-*-... \
                --is-hybrid-model \
                --mamba-state-dim 128 \
                --mamba-head-dim 128 \
                --mamba-num-groups 16 \
                --mamba-num-heads 32
    "
fi
```

### 步骤 2：权重转换

复用 `qwen3_next` 的转换工具：

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

# 创建自定义转换脚本 run_qwen3.5_35b.sh
MODEL_SIZE=35B-A3B
LOAD_DIR=/path/to/Qwen3.5-35B-A3B-HF
SAVE_DIR=/path/to/Qwen3.5-35B-A3B-MCore
MG2HF=false
USE_CUDA=true
PR=bf16

# 复用 scripts/qwen3_next/run_8xH20.sh 的逻辑
# 只需修改传入的 MODEL_SIZE 和路径
```

### 步骤 3：训练启动

```bash
cd /workspace/Pai-Megatron-Patch/examples/qwen3_next

bash run_qwen3.5_35b.sh \
    dlc \
    35B-A3B \
    1 \
    8 \
    1e-5 \
    1e-6 \
    4096 \
    4096 \
    bf16 \
    2 \
    2 \
    1 \
    1 \
    4 \
    true \
    true \
    false \
    false \
    sel \
    false \
    10000 \
    /mnt/datasets/your_data \
    /mnt/datasets/your_data \
    /mnt/ckpts/Qwen3.5-35B-A3B-MCore \
    100000000 \
    10000 \
    /mnt/output
```

## 4. 关键注意事项

### 4.1 Hybrid 架构配置
Qwen3.5-35B-A3B 如果采用 Hybrid (Mamba + Transformer + MoE) 架构，需要确保：
- `--hybrid-override-pattern` 与实际层排列一致
- 检查 `config.json` 中的 `hybrid_config` 字段

### 4.2 序列长度支持
Qwen3.5 可能支持 128K+ 长上下文：
- 确保 `--max-position-embeddings` 设置正确
- 需要开启 Context Parallel (`CP > 1`) 来训练长序列

### 4.3 Tokenizer
复用 Qwen3Tokenizer：
```bash
--patch-tokenizer-type Qwen3Tokenizer
--padded-vocab-size 151936
```

### 4.4 内存优化建议
35B 模型在 8xH20 (80GB) 上的推荐配置：
```bash
TP=2          # Tensor Parallel
PP=2          # Pipeline Parallel  
EP=4          # Expert Parallel
AC=sel        # 选择性重计算
DO=true       # Distributed Optimizer
```

## 5. 验证清单

启动训练前确认：
- [ ] 从 HF 下载的 `config.json` 已正确解析
- [ ] 权重转换成功，MCore 格式 checkpoint 可正常加载
- [ ] `hybrid_override_pattern` 与实际层数匹配
- [ ] 各并行度参数可被整除（`num_layers % PP == 0`, `num_experts % EP == 0`）
- [ ] 数据集已转换为 MMAP 格式（`.bin` + `.idx`）

## 6. 快速启动模板

```bash
# 1. 下载模型
modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir ./Qwen3.5-35B-A3B

# 2. 转换权重
bash convert_qwen3.5_35b.sh bf16

# 3. 启动训练
bash run_qwen3.5_35b.sh dlc 35B-A3B 1 8 1e-5 1e-6 4096 4096 bf16 2 2 1 1 4 true true false false sel false 10000 ./data ./data ./ckpts 100000000 10000 ./output
```

---

**总结**：PAI-Megatron-Patch 已具备支持 Qwen3.5-35B-A3B 的基础设施，只需基于现有 `qwen3_next` 框架，通过**外部脚本传入正确的模型参数**即可，无需修改原仓库代码。
