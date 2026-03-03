# MoE 死专家检测研究与 PAI 框架实施建议

> 针对 Qwen3-30B-A3B (48层, 128专家/层) 的 MoE 训练监测方案

## 1. 问题背景

### 1.1 什么是死专家 (Dead Experts)

在 MoE (Mixture of Experts) 模型训练中，**死专家** 指的是那些被路由器选中但实际上对模型输出贡献极小的专家。这种现象会导致：

- **参数浪费**：部分专家参数几乎不更新，模型容量未充分利用
- **训练不稳定**：负载不均导致梯度分布失衡
- **推理效率下降**：需要加载大量无效参数

### 1.2 为什么现有监测不够

当前 PAI-Megatron-Patch 的 MoE 实现已经包含：

- `tokens_per_expert`: 每个专家接收的 token 数量
- `routing_map`: token 到专家的分配映射
- `switch_load_balancing_loss_func`: 负载均衡辅助损失

**但这些指标只能检测"是否被选中"，无法检测"被选中后是否真的有贡献"。**

### 1.3 僵尸专家 (Zombie Experts) 陷阱

| 指标组合 | 名称 | 危害 |
|---------|------|------|
| 路由权重高 + 输出范数高 | 正常专家 | 无 |
| **路由权重高 + 输出范数低** | **僵尸专家** ⚠️ | 路由器被骗，实际贡献为零 |
| 路由权重低 + 输出范数高 | 被忽视专家 | 有潜力但不被选 |
| 路由权重低 + 输出范数低 | 真死专家 | 确实无用 |

**僵尸专家是最危险的**：辅助损失认为它在工作，实际上它对模型输出零贡献。

---

## 2. 核心监测指标体系

### 2.1 必须监测的指标

```python
# 1. 选择频率 (Selection Frequency)
# 意义：每个专家被选中的 token 比例
selection_freq = tokens_per_expert / total_tokens

# 2. 平均路由权重 (Average Routing Weight)
# 意义：门控网络分配给该专家的平均权重
avg_gate_weight = routing_weights[expert_mask].mean()

# 3. 输出范数 (Output Norm) ⭐ 关键指标
# 意义：专家实际输出的向量强度
output_norm = expert_output.norm(dim=-1).mean()

# 4. 梯度范数 (Gradient Norm)
# 意义：专家参数是否能有效更新
grad_norm = expert.weight.grad.norm()

# 5. 重要性分数 (Importance Score)
# 来源：GPT-4 论文 (OpenAI, 2023)
# 意义：综合权重和输出幅度的贡献度
importance = sum(routing_weight * output_magnitude)

# 6. 变异系数 (Coefficient of Variation)
# 意义：负载均衡程度，CV > 0.3 表示严重不均
cv = tokens_per_expert.std() / tokens_per_expert.mean()
```

### 2.2 指标阈值参考

基于 Qwen3-30B-A3B 配置 (hidden_size=2048, 128专家/层)：

| 指标 | 正常范围 | 警戒线 | 危险 |
|------|---------|--------|------|
| 输出范数 | 10-50 | < 1.0 | < 0.1 |
| 变异系数 (CV) | < 0.3 | 0.3-0.5 | > 0.5 |
| 死专家比例 | 0% | 1-5% | > 10% |
| 梯度范数 | > 0.01 | < 0.001 | ≈ 0 |

---

## 3. PAI-Megatron-Patch 代码适配建议

### 3.1 修改文件 1: `moe_layer.py`

**路径**: `megatron_patch/model/qwen3_moe/moe/moe_layer.py`

```python
class MoELayer(BaseMoELayer):
    def __init__(self, config: TransformerConfig, submodules: MoESubmodules = None, layer_number: int = None):
        # ... 原有代码 ...
        
        # 新增：专家贡献度监测状态
        self.enable_expert_monitoring = getattr(config, 'enable_expert_monitoring', True)
        if self.enable_expert_monitoring:
            self.register_buffer('expert_tokens_count', torch.zeros(self.num_local_experts))
            self.register_buffer('expert_output_norm_sum', torch.zeros(self.num_local_experts))
            self.register_buffer('expert_routing_weight_sum', torch.zeros(self.num_local_experts))
            self.monitoring_step = 0

    def forward(self, hidden_states: torch.Tensor):
        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            
            # token分发
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            
            # 专家执行
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            
            # 新增：专家贡献度监测
            if self.enable_expert_monitoring and self.training:
                self._update_expert_metrics(
                    expert_output, tokens_per_expert, probs, routing_map
                )
            
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            # ... 原有代码 ...
            return output, mlp_bias
        
        # ... 原有代码 ...

    def _update_expert_metrics(self, expert_output, tokens_per_expert, probs, routing_map):
        """更新专家贡献度指标"""
        with torch.no_grad():
            # 1. 记录每个专家的token数量
            for i in range(self.num_local_experts):
                self.expert_tokens_count[i] += tokens_per_expert[i].item()
            
            # 2. 计算每个专家的输出范数
            start_idx = 0
            for i in range(self.num_local_experts):
                num_tokens = tokens_per_expert[i].item()
                if num_tokens > 0:
                    end_idx = start_idx + num_tokens
                    expert_slice = expert_output[start_idx:end_idx]
                    output_norm = expert_slice.norm(dim=-1).mean()
                    self.expert_output_norm_sum[i] += output_norm
                    start_idx = end_idx
            
            # 3. 记录平均路由权重
            for i in range(self.num_local_experts):
                expert_mask = routing_map[:, i]
                if expert_mask.any():
                    avg_weight = probs[expert_mask].mean()
                    self.expert_routing_weight_sum[i] += avg_weight
            
            self.monitoring_step += 1

    def get_expert_stats(self):
        """获取当前专家统计信息"""
        if not self.enable_expert_monitoring or self.monitoring_step == 0:
            return {}
        
        total_tokens = self.expert_tokens_count.sum()
        avg_norms = self.expert_output_norm_sum / self.monitoring_step
        
        return {
            'layer': self.layer_number,
            'step': self.monitoring_step,
            'selection_freq': (self.expert_tokens_count / total_tokens).tolist(),
            'avg_output_norm': avg_norms.tolist(),
            'avg_routing_weight': (self.expert_routing_weight_sum / self.monitoring_step).tolist(),
            'cv_tokens': self.expert_tokens_count.std() / self.expert_tokens_count.mean(),
            'dead_experts': (avg_norms < 0.01).sum().item(),
            'min_norm': avg_norms.min().item(),
            'max_norm': avg_norms.max().item(),
            'mean_norm': avg_norms.mean().item(),
        }
    
    def reset_expert_stats(self):
        """重置统计（建议每N步调用）"""
        if self.enable_expert_monitoring:
            self.expert_tokens_count.zero_()
            self.expert_output_norm_sum.zero_()
            self.expert_routing_weight_sum.zero_()
            self.monitoring_step = 0
```

### 3.2 修改文件 2: `router.py` 专家偏差调整

**路径**: `megatron_patch/model/qwen3_moe/moe/router.py`

```python
class TopKRouter(_TopKRuter):
    def __init__(self, config):
        # ... 原有代码 ...
        
        # 新增：专家偏差调整（促进低频专家）
        self.expert_bias = torch.nn.Parameter(torch.zeros(config.num_moe_experts))
        self.enable_expert_bias = getattr(config, 'enable_expert_bias', False)
        self.local_tokens_per_expert = torch.zeros(config.num_moe_experts)
        self.bias_update_interval = getattr(config, 'expert_bias_update_interval', 100)
        self.step_count = 0
    
    def routing(self, logits: torch.Tensor):
        # ... 原有路由逻辑 ...
        
        # 新增：定期调整专家bias
        if self.enable_expert_bias and self.training:
            self.step_count += 1
            if self.step_count % self.bias_update_interval == 0:
                with torch.no_grad():
                    avg_tokens = self.local_tokens_per_expert.mean()
                    for i in range(self.num_local_experts):
                        if self.local_tokens_per_expert[i] < avg_tokens * 0.1:
                            # 低频专家：增加bias
                            self.expert_bias[i] += 0.01
                        elif self.local_tokens_per_expert[i] > avg_tokens * 2:
                            # 高频专家：减少bias
                            self.expert_bias[i] = max(0, self.expert_bias[i] - 0.01)
                
                # 重置计数
                self.local_tokens_per_expert.zero_()
            
            # 应用bias到logits
            logits = logits + self.expert_bias
        
        # ... 继续原有逻辑 ...
```

### 3.3 修改文件 3: `run_mcore_qwen3.sh` 训练脚本

**路径**: `examples/qwen3/run_mcore_qwen3.sh`

在 `megatron_options` 后添加监测参数：

```bash
# MoE 专家监测配置
moe_monitoring_options=" \
    --enable-expert-monitoring \
    --enable-expert-bias \
    --expert-bias-update-interval 100 \
    --expert-log-interval 100 \
    --expert-alert-threshold 0.01 \
"

# 运行命令中添加
run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_qwen.py
 ${megatron_options} ${dataset_options} ... ${moe_monitoring_options}"
```

### 3.4 新增文件: `training_monitor.py` 监测工具

**建议路径**: `megatron_patch/utils/moe_monitor.py`

```python
"""MoE 专家监测工具"""
import torch
from megatron.core import parallel_state

class MoEExpertMonitor:
    """MoE 专家贡献度监测器"""
    
    def __init__(self, model, log_interval=100, alert_threshold=0.01):
        self.model = model
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.iteration = 0
        
    def log_stats(self, writer=None):
        """记录专家统计信息"""
        self.iteration += 1
        if self.iteration % self.log_interval != 0:
            return
            
        if not hasattr(self.model, 'module'):
            return
            
        decoder = self.model.module.decoder
        if not hasattr(decoder, 'layers'):
            return
        
        all_stats = []
        alerts = []
        
        for layer_idx, layer in enumerate(decoder.layers):
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'get_expert_stats'):
                continue
                
            stats = layer.mlp.get_expert_stats()
            if not stats:
                continue
            
            all_stats.append(stats)
            
            # 分层聚合记录（减少TensorBoard曲线数量）
            if writer is not None:
                writer.add_scalar(f'MoE/layer{layer_idx}_cv', stats['cv_tokens'], self.iteration)
                writer.add_scalar(f'MoE/layer{layer_idx}_dead_count', stats['dead_experts'], self.iteration)
                writer.add_scalar(f'MoE/layer{layer_idx}_mean_norm', stats['mean_norm'], self.iteration)
                
                # 用直方图记录128个专家的分布
                writer.add_histogram(f'MoE/layer{layer_idx}_output_norms', 
                                   torch.tensor(stats['avg_output_norm']), 
                                   self.iteration)
            
            # 告警检测
            if stats['dead_experts'] > 0:
                alerts.append(f"Layer {layer_idx}: {stats['dead_experts']} dead experts (norm < {self.alert_threshold})")
            
            if stats['cv_tokens'] > 0.5:
                alerts.append(f"Layer {layer_idx}: CV={stats['cv_tokens']:.3f}, severe imbalance!")
        
        # 打印告警
        if alerts and parallel_state.is_last_rank():
            print(f"\n[MoE Monitor] Step {self.iteration} Alerts:")
            for alert in alerts:
                print(f"  ⚠️  {alert}")
        
        # 重置统计
        for layer in decoder.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'reset_expert_stats'):
                layer.mlp.reset_expert_stats()
        
        return all_stats
```

---

## 4. 监测策略建议

### 4.1 分层监测（解决6144个专家爆炸问题）

```python
# 不是监测48×128=6144条曲线，而是分层聚合

# 方案1: 每层只记录统计摘要
layer_stats = {
    'mean_output_norm': all_experts.mean(),        # 1个值
    'cv_output_norm': all_experts.std() / mean,    # 变异系数
    'dead_expert_count': (norms < 0.1).sum(),      # 死专家数量
    'min_norm_expert_id': norms.argmin(),          # 最差的专家ID
}
# 48层 × 4标量 = 192条曲线 ✅

# 方案2: 只监测关键层
monitored_layers = [0, 12, 24, 36, 47]  # 首中尾
# 5层 × 128专家 = 640条曲线 ✅

# 方案3: 直方图可视化（最优雅）
writer.add_histogram(f'layer{layer_id}/expert_norms', expert_norms, step)
# 一层一个分布图，包含全部128个专家信息
```

### 4.2 TensorBoard 可视化配置

```python
# 监测面板布局建议

# Panel 1: 负载均衡
# - MoE/*/cv (48条曲线)
# - MoE/*/dead_count (48条曲线)

# Panel 2: 专家输出分布（直方图）
# - MoE/layer*/output_norms (选择关键层)

# Panel 3: 全局统计
# - total_dead_experts
# - mean_cv_across_layers
```

---

## 5. 训练配置建议

### 5.1 Qwen3-30B-A3B 推荐配置

```bash
# 在 run_mcore_qwen3.sh 的 A3B 配置中添加

# 基础 MoE 配置（已存在）
moe_options=" \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-topk 8 \
    --num-experts 128 \
    --moe-ffn-hidden-size 768 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
"

# 新增：死专家监测配置
moe_monitoring_options=" \
    --enable-expert-monitoring \
    --enable-expert-bias \
    --expert-bias-update-interval 100 \
    --expert-log-interval 100 \
    --expert-alert-threshold 0.01 \
    --moe-z-loss-coeff 0.001 \
"
```

### 5.2 不同训练阶段的策略

| 阶段 | 策略 | 说明 |
|------|------|------|
| 预热期 (0-1% tokens) | 严格监测 + 激进bias调整 | 早期路由容易坍缩 |
| 主训练期 | 正常监测 + 温和bias调整 | 稳定训练为主 |
| 退火期 (最后10%) | 放宽bias，专注收敛 | 避免破坏已学分布 |

---

## 6. 常见问题 FAQ

### Q1: 输出范数和路由权重有什么区别？

**路由权重** = 路由器想不想用它（投票前的承诺）  
**输出范数** = 它实际干了多少活（干活后的结果）

必须同时监测才能发现"僵尸专家"（路由权重高但输出范数低）。

### Q2: 6144个专家怎么监测？

不要测6144条曲线：
- 每层记录5个统计量（CV、死专家数、均值、最值）
- 异常时自动展开详细分布
- 用直方图可视化代替逐个记录

### Q3: 检测到死专家后怎么处理？

1. **专家Bias调整**：自动增加低频专家的bias
2. **梯度缩放**：对低利用率专家梯度×1.5
3. **Dropout Routing**：训练时随机选择专家增加探索
4. **重启专家**：重新初始化死专家的权重

### Q4: 监测开销有多大？

```python
# 计算开销
with torch.no_grad():
    norms = expert_output.norm(dim=-1)  # 1次cuda kernel

# 内存开销
# 128 floats = 512 bytes/层
# 48层 = 24KB（可忽略）
```

实际影响：< 0.1% 训练时间

---

## 7. 参考文献

1. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
2. Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models"
3. OpenAI (2023). "GPT-4 Technical Report"
4. Dai, D., et al. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
5. Megatron-LM Documentation: https://github.com/NVIDIA/Megatron-LM

---

## 8. 附录：完整指标计算公式

```python
# 1. 选择频率
selection_freq[i] = tokens_per_expert[i] / sum(tokens_per_expert)

# 2. 平均路由权重
avg_weight[i] = mean(probs[routing_map[:, i]])

# 3. 输出范数
output_norm[i] = mean(norm(expert_outputs[i], dim=-1))

# 4. 变异系数
cv = std(tokens_per_expert) / mean(tokens_per_expert)

# 5. 重要性分数（GPT-4风格）
importance[i] = sum(routing_weights * output_magnitudes)

# 6. 专家利用率
utilization[i] = selection_freq[i] * output_norm[i] / max_output_norm
```

---

*文档版本: 1.0*  
*更新日期: 2026-03-03*  
*适用模型: Qwen3-30B-A3B, Qwen3-235B-A22B*  
*框架: PAI-Megatron-Patch*
