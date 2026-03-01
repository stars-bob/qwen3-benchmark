#!/usr/bin/env python3
"""
Qwen3-1.7B 精度对比测试
PyTorch FP16 vs MLX bf16
"""

import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load as mlx_load, generate as mlx_generate


def load_pytorch_model():
    """加载 PyTorch FP16 模型"""
    print("📥 加载 PyTorch FP16 模型...")
    model_name = "Qwen/Qwen3-1.7B"
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device.type == "mps" else None,
        trust_remote_code=True
    )
    if device.type == "cpu":
        model = model.to(device)
    
    return model, tokenizer, device


def load_mlx_model():
    """加载 MLX bf16 模型"""
    print("📥 加载 MLX bf16 模型...")
    model_name = "Qwen/Qwen3-1.7B-MLX-bf16"
    model, tokenizer = mlx_load(model_name)
    return model, tokenizer


def generate_pytorch(model, tokenizer, device, prompt, max_tokens=256):
    """PyTorch 生成"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  #  greedy decoding for consistency
        )
    gen_time = time.time() - start
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, gen_time


def generate_mlx(model, tokenizer, prompt, max_tokens=256):
    """MLX 生成"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    start = time.time()
    response = mlx_generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
    gen_time = time.time() - start
    
    return response, gen_time


def run_comparison():
    """运行对比测试"""
    
    # 测试用例设计：覆盖不同能力维度
    test_cases = [
        {
            "name": "数学计算",
            "prompt": "计算：1234 × 5678 = ? 请展示计算过程。",
            "category": "math"
        },
        {
            "name": "逻辑推理",
            "prompt": "有三个盒子，分别标有'苹果'、'橙子'、'混合'，但所有标签都贴错了。你只能从一个盒子里拿出一个水果查看，如何确定每个盒子的真实内容？",
            "category": "logic"
        },
        {
            "name": "代码生成",
            "prompt": "用 Python 实现一个 LRU 缓存，要求 O(1) 时间复杂度，并包含简单的使用示例。",
            "category": "code"
        },
        {
            "name": "事实知识",
            "prompt": "《三体》的作者是谁？这部作品获得了什么奖项？请简要介绍主要内容。",
            "category": "knowledge"
        },
        {
            "name": "复杂推理",
            "prompt": "如果一艘飞船以 0.9 倍光速从地球飞向 4.3 光年外的比邻星，飞船上的人感觉经过了多长时间？请解释狭义相对论的时间膨胀效应。",
            "category": "physics"
        },
        {
            "name": "中文理解",
            "prompt": "解释这句古诗的含义：'落霞与孤鹜齐飞，秋水共长天一色。' 作者是谁？表达了什么意境？",
            "category": "chinese"
        },
    ]
    
    print("=" * 80)
    print("🧪 Qwen3-1.7B 精度对比测试")
    print("   PyTorch FP16 (transformers + MPS) vs MLX bf16")
    print("=" * 80)
    
    # 加载两个模型
    pt_model, pt_tokenizer, pt_device = load_pytorch_model()
    mlx_model, mlx_tokenizer = load_mlx_model()
    
    print("\n" + "=" * 80)
    print("开始测试...")
    print("=" * 80)
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"【测试 {i}/{len(test_cases)}】{test['name']} ({test['category']})")
        print(f"Prompt: {test['prompt']}")
        print('─' * 80)
        
        # PyTorch 生成
        print("\n🔵 PyTorch FP16:")
        pt_response, pt_time = generate_pytorch(pt_model, pt_tokenizer, pt_device, test['prompt'])
        print(f"⏱️  {pt_time:.2f}s")
        print(f"📝 {pt_response[:500]}{'...' if len(pt_response) > 500 else ''}")
        
        # MLX 生成
        print("\n🟣 MLX bf16:")
        mlx_response, mlx_time = generate_mlx(mlx_model, mlx_tokenizer, test['prompt'])
        print(f"⏱️  {mlx_time:.2f}s")
        print(f"📝 {mlx_response[:500]}{'...' if len(mlx_response) > 500 else ''}")
        
        results.append({
            "test": test['name'],
            "category": test['category'],
            "pytorch_time": pt_time,
            "mlx_time": mlx_time,
            "pytorch_response": pt_response,
            "mlx_response": mlx_response
        })
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 测试结果总结")
    print("=" * 80)
    
    print(f"\n{'测试项':<12} {'类别':<10} {'PyTorch':<12} {'MLX':<12} {'速度比':<10}")
    print("-" * 60)
    
    total_pt_time = 0
    total_mlx_time = 0
    
    for r in results:
        speedup = r['pytorch_time'] / r['mlx_time'] if r['mlx_time'] > 0 else 0
        print(f"{r['test']:<12} {r['category']:<10} {r['pytorch_time']:.2f}s{'':<6} {r['mlx_time']:.2f}s{'':<6} {speedup:.2f}x")
        total_pt_time += r['pytorch_time']
        total_mlx_time += r['mlx_time']
    
    print("-" * 60)
    print(f"{'总计':<12} {'':<10} {total_pt_time:.2f}s{'':<6} {total_mlx_time:.2f}s{'':<6} {total_pt_time/total_mlx_time:.2f}x")
    
    print("\n" + "=" * 80)
    print("💡 观察要点：")
    print("   1. 数学计算：两者结果是否一致？计算过程是否正确？")
    print("   2. 代码生成：语法是否正确？逻辑是否完整？")
    print("   3. 事实知识：准确性是否有差异？")
    print("   4. 推理深度：分析是否深入？逻辑是否清晰？")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_comparison()
