#!/usr/bin/env python3
"""
Qwen3-1.7B-Instruct 模型推理测试
适配 Apple Silicon Mac (MPS 加速)
"""

import os
# 设置 Hugging Face 镜像（国内加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_device():
    """检查可用设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ 使用 MPS (Metal Performance Shaders) 加速")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print(f"⚠️ 使用 CPU (速度较慢)")
    return device

def main():
    model_name = "Qwen/Qwen3-1.7B"
    
    print(f"🤖 正在加载模型: {model_name}")
    print("-" * 50)
    
    # 检查设备
    device = check_device()
    print("-" * 50)
    
    # 加载 tokenizer
    print("📥 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 加载模型
    print("📥 加载模型 (可能需要几分钟)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        device_map="auto" if device.type == "mps" else None,
        trust_remote_code=True
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    print("✅ 模型加载完成！")
    print("-" * 50)
    
    # 测试推理
    test_prompts = [
        "你好，请介绍一下自己。",
        "What is the capital of France?",
        "用Python写一个快速排序算法。",
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 输入: {prompt}")
        print("🤖 输出:", end=" ")
        
        # Qwen3 使用 chat 模板
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 解码输出
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        print(response)
        print("-" * 50)
    
    print("\n✅ 所有测试完成！")
    
    # 交互模式
    print("\n💬 进入交互模式 (输入 'exit' 退出):")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("👋 再见！")
            break
        
        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        print(f"🤖: {response}")

if __name__ == "__main__":
    main()
