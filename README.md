# Qwen3 Benchmark Repository

Qwen3 系列模型的架构分析、训练研究与基准测试仓库。

---

## ⚠️ 重要更新 (2026-03-03)

**仓库重构完成**: 所有文档已按类别整理到对应目录，新增 MoE 死专家检测研究。

---

## 目录结构

```
qwen3-benchmark/
├── README.md                                    # 本文件
├── docs/
│   ├── architecture/                            # 📐 架构分析
│   │   ├── HF-Config-Comparison-Qwen3.5-vs-Qwen3-Next.md
│   │   └── Qwen3.5-35B-A3B-Architecture-Confirmed.md
│   └── training-research/                       # 🏋️ 训练研究
│       └── MoE-Dead-Expert-Detection-Research-and-Implementation.md
```

---

## 文档分类

### 📁 docs/architecture/ - 架构分析

包含模型的配置对比、架构确认和技术差异分析。

| 文件 | 内容描述 |
|------|----------|
| `HF-Config-Comparison-Qwen3.5-vs-Qwen3-Next.md` | Qwen3.5-35B-A3B 与 Qwen3-Next-80B-A3B 的 HuggingFace 配置详细对比 |
| `Qwen3.5-35B-A3B-Architecture-Confirmed.md` | Qwen3.5-35B-A3B 架构参数确认与 Linear Attention 分析 |

**关键发现**: Qwen3-Next 和 Qwen3.5 都使用 **Linear Attention**（不是 Mamba），架构相似度高达 95%。

### 📁 docs/training-research/ - 训练研究

包含训练过程中的技术研究与优化方案。

| 文件 | 内容描述 |
|------|----------|
| `MoE-Dead-Expert-Detection-Research-and-Implementation.md` | MoE 模型死专家检测研究与 PAI-Megatron-Patch 实施建议 |

**核心内容**: 
- 6个核心监测指标（输出范数、路由权重、变异系数、重要性分数等）
- PAI-Megatron-Patch 三处代码修改方案
- 48层×128专家（6144个）的监测策略
- 专家唤醒与负载均衡优化

---

## 快速导航

- **架构差异对比** → `docs/architecture/HF-Config-Comparison-Qwen3.5-vs-Qwen3-Next.md`
- **架构参数确认** → `docs/architecture/Qwen3.5-35B-A3B-Architecture-Confirmed.md`
- **MoE死专家检测** → `docs/training-research/MoE-Dead-Expert-Detection-Research-and-Implementation.md`

---

## 模型对比速览

| 特性 | Qwen3-Next-80B-A3B | Qwen3.5-35B-A3B |
|------|-------------------|-----------------|
| **架构** | Linear Attention + MoE | Linear Attention + MoE |
| **层数 (HF)** | 48 | **40** |
| **专家数** | 512 | **256** |
| **TopK** | 10 | **8** |
| **词表大小** | 151936 | **248320** (+63%) |
| **MTP** | ❌ 否 | ✅ 是 |
| **多模态** | ❌ 否 | ✅ 是 |

---

## 相关仓库

- [PAI-Megatron-Patch](https://github.com/alibaba/PAI-Megatron-Patch) - 阿里云 PAI 的 Megatron 训练补丁
- [ms-swift](https://github.com/modelscope/ms-swift) - ModelScope 轻量训练框架

---

## 更新日志

### 2026-03-03
- **仓库重构**: 文档按类别整理到 `docs/architecture/` 和 `docs/training-research/`
- **新增研究**: MoE 死专家检测与 PAI 框架实施建议

### 2026-03-02
- 架构修正：确认 Linear Attention 架构（非 Mamba）
- 新增 ms-swift 框架支持研究
- 新增 PAI vs ms-swift 对齐验证研究

---

**最后更新**: 2026-03-03  
**维护者**: stars-bob
