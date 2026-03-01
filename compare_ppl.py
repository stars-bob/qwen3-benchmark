#!/usr/bin/env python3
"""
Qwen3-1.7B Perplexity (PPL) 对比测试
PyTorch FP16 vs MLX bf16
"""

import os
import math
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import mlx.core as mx
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load as mlx_load


def calculate_ppl_pytorch(model, tokenizer, text, device):
    """计算 PyTorch 模型的 PPL"""
    # 编码文本
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        # 获取 logits
        outputs = model(input_ids)
        logits = outputs.logits
        
        # 计算 loss (shift logits and labels)
        # logits[:, :-1, :] 预测下一个 token
        # input_ids[:, 1:] 是目标
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # 计算 cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        ppl = math.exp(loss.item())
    
    return ppl, loss.item()


def calculate_ppl_mlx(model, tokenizer, text):
    """计算 MLX 模型的 PPL"""
    # 编码文本
    input_ids = mx.array(tokenizer.encode(text, return_tensors="np"))
    
    # 前向传播
    logits = model(input_ids)
    logits = logits[0]  # shape: (seq_len, vocab_size)
    
    # 使用 MLX 操作计算 log softmax
    log_probs = mx.log(mx.softmax(logits, axis=-1))
    
    # 获取目标 token 的 log prob
    # input_ids[1:] 是目标，对应 logits[:-1] 的预测
    target_ids = input_ids[0, 1:]
    pred_logits = log_probs[:-1, :]
    
    # 收集目标 token 的 log prob
    target_log_probs = pred_logits[mx.arange(len(target_ids)), target_ids]
    
    # 计算平均 negative log likelihood
    nll = -mx.mean(target_log_probs).item()
    ppl = math.exp(nll)
    
    return ppl, nll


def run_ppl_comparison():
    """运行 PPL 对比测试"""
    
    # 测试文本 - 选择不同类型的内容
    test_texts = [
        {
            "name": "中文新闻",
            "text": "据新华社报道，中国科学家近日在量子计算领域取得重大突破，成功研制出新型量子处理器，运算速度比传统超级计算机快一亿倍。"
        },
        {
            "name": "英文科技",
            "text": "Artificial intelligence has revolutionized numerous industries, from healthcare to autonomous driving. Large language models demonstrate remarkable capabilities in understanding and generating human-like text."
        },
        {
            "name": "代码片段",
            "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Example usage\nprint([fibonacci(i) for i in range(10)])"
        },
        {
            "name": "学术论文",
            "text": "The attention mechanism allows the model to focus on different parts of the input sequence when producing each element of the output sequence. This is particularly useful in machine translation tasks."
        }
    ]
    
    print("=" * 80)
    print("🧮 Perplexity (PPL) 对比测试")
    print("   PyTorch FP16 vs MLX bf16")
    print("=" * 80)
    print("\n⚠️  注意：PPL 越低越好（表示模型对文本的预测越准确）")
    print("-" * 80)
    
    # 加载 PyTorch 模型
    print("\n📥 加载 PyTorch FP16 模型...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pt_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    pt_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "mps" else None,
        trust_remote_code=True
    )
    if device.type == "cpu":
        pt_model = pt_model.to(device)
    pt_model.eval()
    print("✅ PyTorch 模型加载完成")
    
    # 加载 MLX 模型
    print("\n📥 加载 MLX bf16 模型...")
    mlx_model, mlx_tokenizer = mlx_load("Qwen/Qwen3-1.7B-MLX-bf16")
    print("✅ MLX 模型加载完成")
    
    print("\n" + "=" * 80)
    print("开始计算 PPL...")
    print("=" * 80)
    
    results = []
    
    for test in test_texts:
        print(f"\n【{test['name']}】")
        print(f"文本长度: {len(test['text'])} 字符")
        
        # PyTorch PPL
        start = time.time()
        pt_ppl, pt_loss = calculate_ppl_pytorch(pt_model, pt_tokenizer, test['text'], device)
        pt_time = time.time() - start
        
        # MLX PPL
        start = time.time()
        mlx_ppl, mlx_loss = calculate_ppl_mlx(mlx_model, mlx_tokenizer, test['text'])
        mlx_time = time.time() - start
        
        # 计算差异
        ppl_diff = abs(pt_ppl - mlx_ppl)
        ppl_diff_pct = (ppl_diff / pt_ppl) * 100 if pt_ppl > 0 else 0
        
        print(f"  🔵 PyTorch FP16: PPL={pt_ppl:.4f}, Loss={pt_loss:.4f}, Time={pt_time:.3f}s")
        print(f"  🟣 MLX bf16:    PPL={mlx_ppl:.4f}, Loss={mlx_loss:.4f}, Time={mlx_time:.3f}s")
        print(f"  📊 差异: {ppl_diff:.4f} ({ppl_diff_pct:.2f}%)")
        
        results.append({
            "name": test['name'],
            "pt_ppl": pt_ppl,
            "mlx_ppl": mlx_ppl,
            "pt_loss": pt_loss,
            "mlx_loss": mlx_loss,
            "diff": ppl_diff,
            "diff_pct": ppl_diff_pct
        })
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 PPL 对比总结")
    print("=" * 80)
    
    print(f"\n{'测试项':<15} {'PyTorch PPL':<15} {'MLX PPL':<15} {'差异':<12} {'差异%':<10}")
    print("-" * 70)
    
    total_diff = 0
    for r in results:
        print(f"{r['name']:<15} {r['pt_ppl']:<15.4f} {r['mlx_ppl']:<15.4f} {r['diff']:<12.4f} {r['diff_pct']:<10.2f}%")
        total_diff += r['diff_pct']
    
    avg_diff = total_diff / len(results)
    print("-" * 70)
    print(f"{'平均差异':<15} {'':<15} {'':<15} {'':<12} {avg_diff:<10.2f}%")
    
    print("\n" + "=" * 80)
    if avg_diff < 1.0:
        print("✅ 结论：两个模型的 PPL 差异极小 (<1%)，精度基本一致")
    elif avg_diff < 5.0:
        print("⚠️  结论：PPL 有轻微差异 (1-5%)，但仍在可接受范围")
    else:
        print("❌ 结论：PPL 差异较大 (>5%)，精度可能存在差异")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import numpy as np  # MLX 计算需要
    run_ppl_comparison()
