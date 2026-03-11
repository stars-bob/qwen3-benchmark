# Scalable Training of Mixture-of-Experts Models with Megatron Core - 实验设计

## 1. 基准测试模型

### 1.1 DeepSeek-V3-685B
- **总参数量**: 685B
- **专家数量**: 256个
- **激活专家**: top-8 路由
- **架构特点**: 
  - 细粒度专家设计 (fine-grained MoE)
  - Multi-Token Prediction (MTP)
  - Multi-Latent Attention (MLA)
  - 61个decoder层

### 1.2 Qwen3-235B-A22B
- **总参数量**: 235B
- **激活参数量**: 22B
- **架构特点**: 细粒度MoE架构

---

## 2. 硬件平台

### 2.1 NVIDIA GB300 / GB200
- **显存**: 192GB per GPU
- **互联**: NVL72 (72 GPU NVLink domain)
- **特性**: 
  - Native FP8 Tensor Core支持
  - NVLink-C2C高带宽
  - 1.8 TB/s双向带宽

### 2.2 NVIDIA H100
- **显存**: 80GB per GPU
- **互联**: NVL8 (8 GPU NVLink domain)
- **规模**: 1024 GPUs (用于DeepSeek-V3测试)

---

## 3. 并行策略设计

### 3.1 DeepSeek-V3 配置对比

| 配置项 | GB200 | H100 |
|--------|-------|------|
| 硬件规模 | 256×GB200 | 1024×H100 |
| TP/PP/EP | 1/4/64 | 2/8/64 |
| VPP | 4 | 4 |
| GBS/MBS/SeqLen | 8192/1/4096 | 8192/1/4096 |
| 精度 | MXFP8 | FP8-Blockwise |
| Token分发器 | HybridEP | DeepEP |
| 重计算 | mlp | mlp, mla_up_proj, moe_act, layernorm |
| CUDA Graphs | Enabled | — |
| EP all-to-all重叠 | — | Enabled |
| **性能** | **1,048 TFLOPS/GPU** | **368 TFLOPS/GPU** |

### 3.2 Parallel Folding设计
- **核心思想**: 注意力层和专家层使用独立的并行策略
- **注意力层**: 使用TP减少内存
- **专家层**: EP=1，每个专家运行在单个GPU上，最大化GEMM效率
- **EP64选择**: 每个GPU持有4个专家(256/64=4)，消除本地token置换开销

### 3.3 Flexible Asymmetric VPP (虚拟流水线并行)

**DeepSeek-V3布局 (PP=16, VPP=2)**:
- **PP Rank 0**: embedding + 3×decoder | 2×decoder
- **PP Rank 1-13**: 2×decoder | 2×decoder
- **PP Rank 14**: 2×decoder | MTP
- **PP Rank 15**: 2×decoder | loss

**目的**: 平衡不同层的计算成本 (MoE层 vs 密集层 vs Embedding/Loss)

---

## 4. 内存优化实验

### 4.1 FP8训练实验
- **效果**: 激活内存减半
- **GB200**: 使用MXFP8，native Blackwell Tensor Core支持
- **H100**: 使用FP8-Blockwise精度

### 4.2 细粒度重计算策略对比

**GB200配置**:
```
--recompute-modules mlp
```
- 仅重计算MLP层
- 原因: NVL72本地EP无需通信重叠，节省内存

**H100配置**:
```
--recompute-modules mlp,mla_up_proj,moe_act,layernorm
```
- 更激进的重计算策略
- 原因: 需要为EP通信重叠预留额外缓冲区

### 4.3 优化器状态优化
- **低精度优化器**: BF16优化器状态
- **优化器卸载 (GB200)**: 
  - 利用NVLink-C2C高带宽
  - 释放大量GPU内存用于激活
  - 减少重计算需求

### 4.4 内存高效置换
- 消除冗余激活存储
- 原地(in-place)操作优化

---

## 5. 通信优化实验

### 5.1 Token分发器对比

**HybridEP (GB200)**:
- 针对NVL72优化
- 充分利用1.8 TB/s NVLink带宽
- 无需通信重叠

**DeepEP (H100)**:
- 融合置换(fused permutation)
- 优化的集合通信操作
- 必须配合EP通信重叠

### 5.2 EP all-to-all重叠实验 (H100)
```
--overlap-moe-expert-parallel-comm
```
- **目的**: 隐藏跨节点延迟
- **机制**: 流水线化dispatch和combine操作
- **前提**: FP8释放足够内存用于额外缓冲区

### 5.3 通信墙诊断实验
- **症状**: 分析显示集合通信占用30-50%的step时间
- **解决方案**: DeepEP + EP重叠

---

## 6. 计算效率优化实验

### 6.1 CUDA Graphs实验 (GB200)
```
--cuda-graph-impl transformer_engine
```
- **部分捕获**: 
  - 捕获: Attention, Router, MoE预处理
  - 不捕获: 动态专家计算
- **效果**: 消除CPU开销瓶颈
- **副作用**: 消耗额外内存

### 6.2 算子融合实验
- **Router融合**: `--moe-router-fusion`
- **置换融合**: `--moe-permute-fusion`
- **MLA RoPE融合**: 减少kernel启动次数

### 6.3 Grouped GEMM实验
```
--moe-grouped-gemm
```
- 目的: 提高小GEMM的GPU利用率
- 解决细粒度MoE的GEMM效率问题

### 6.4 CPU/NUMA绑定
- 使用`bindpcie`脚本
- 基于local rank自动检测GPU/NUMA拓扑
- 使用numactl绑定CPU和内存到最近NUMA节点

---

## 7. 长上下文训练实验

### 7.1 Dynamic-CP (动态Context Parallelism)
- **场景**: 可变长度序列 (THD格式)
- **策略**: 根据每个microbatch的序列长度动态选择cp_size
- **效果**: 在imbalanced序列长度场景下实现35-60%性能提升

### 7.2 序列并行策略实验
**测试配置**: 256 Hopper GPUs @ 256K序列长度

| 模型 | 优化策略 | 相对于短上下文MFU |
|------|----------|-------------------|
| DeepSeek-V3 | TP + Optimizer CPU Offloading + Selective Recompute | 88% |
| Qwen3-235B | TP + CP + Selective Recompute | 129% |

**Qwen3超过100%原因**: SDPA kernel在长序列时计算效率极高

### 7.3 Packed Sequences实验
- **目的**: 消除padding开销
- **格式**: THD (Total-Length, Header, Data)
- **负载均衡**: 基于"attention cost" (序列长度平方)排序
- **调度策略**: 小-大-小(蛇形)模式

---

## 8. 完整性能基准测试

### 8.1 主要性能结果 (Table 11)

| 模型 | 硬件 | 精度 | 序列长度 | 并行配置 | TFLOPS/GPU |
|------|------|------|----------|----------|------------|
| DeepSeek-V3-685B | GB300 | FP8 | 4096 | TP1/PP4/EP64 | **1,233** |
| DeepSeek-V3-685B | GB200 | FP8 | 4096 | TP1/PP4/EP64 | **1,048** |
| DeepSeek-V3-685B | H100 | FP8-Blockwise | 4096 | TP2/PP8/EP64 | **368** |
| Qwen3-235B-A22B | GB300 | FP8 | 4096 | TP1/PP2/EP16 | **974** |
| Qwen3-235B-A22B | GB200 | FP8 | 4096 | TP1/PP2/EP16 | **919** |

### 8.2 FP8训练有效性验证
- H100上DeepSeek-V3达到368 TFLOPS/GPU
- 证明blockwise FP8 recipe可以在Hopper架构上高效训练大规模MoE

---

## 9. 优化流程验证实验 (Three-Phase Workflow)

### Phase 1: 内存可行性
- **工具**: Memory Estimator GUI
- **方法**: 使用`--fake-init-process-group`在单GPU上快速测试
- **案例**: 685B模型BF16激活单独超过130GB，排除80GB设备基线配置

### Phase 2: 并行策略选择
- **指南验证**: 
  - 最小化TP/EP/PP/CP，最大化DP
  - EP×TP保持在NVLink域内
  - PP用于跨节点扩展

### Phase 3: 瓶颈分析与优化
**Memory Wall诊断**:
- 症状: 被迫使用全重计算或过大并行度
- 解决: FP8 + 选择性重计算 + 卸载

**Communication Wall诊断**:
- 症状: 分析显示集合通信占主导
- 解决: 根据通信类型选择优化策略

**Compute Efficiency Wall诊断**:
- CPU开销: Nsight Systems显示kernel间隙
- 计算低效: GPU SM利用率低

---

## 10. RL (强化学习) 实验设计

### 10.1 RL后训练挑战
1. **可变长度序列**: 最大128K-1M tokens，均值仅为最大值的1/4-1/2
2. **内存卸载**: 训练和推理引擎共享GPU，需要快速状态切换
3. **在线权重导出**: 每步训练后快速更新推理引擎
4. **训练稳定性**: 推理和训练引擎使用不同优化kernel导致的路由差异

### 10.2 实验配置
- **框架**: Megatron-Core运行在Ray workers中
- **转换**: Megatron-Bridge用于HF↔Megatron格式转换
- **优化**: Packed Sequence + 负载均衡策略

---

## 11. 关键实验结论

### 11.1 平台特性决定优化策略
- **GB200优势**: 大显存(192GB) + 高C2C带宽 → 更激进的选择
- **H100挑战**: NVL8需要EP通信重叠来隐藏跨节点延迟

### 11.2 Parallel Folding的必要性
- 解耦attention TP和expert EP
- 允许独立优化每种层类型
- 配合Flexible VPP实现细粒度负载均衡

### 11.3 FP8的副作用
- 优点: 减少内存 + 加速GEMM
- 缺点: 放大CPU开销 → 需要CUDA Graphs

### 11.4 迭代优化必要性
- 内存优化 → 释放内存 → 启用通信重叠
- 通信优化 → 暴露计算效率瓶颈
- 持续分析指导下一轮优化目标
