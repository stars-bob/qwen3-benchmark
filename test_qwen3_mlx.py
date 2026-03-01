#!/usr/bin/env python3
"""
Qwen3-1.7B-MLX-bf16 模型推理测试
Apple Silicon 原生 MLX 框架，全精度 bf16
"""

import os
import time

# 设置 Hugging Face 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mlx_lm import load, generate

def main():
    model_name = "Qwen/Qwen3-1.7B-MLX-bf16"
    
    print(f"🤖 正在加载 MLX 模型: {model_name}")
    print("-" * 50)
    
    # 加载模型和 tokenizer
    start_time = time.time()
    model, tokenizer = load(model_name)
    load_time = time.time() - start_time
    print(f"✅ 模型加载完成！耗时: {load_time:.2f} 秒")
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
        
        # 应用 chat 模板
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 记录生成时间
        gen_start = time.time()
        
        # 生成回复
        response = generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=256,
            verbose=False
        )
        
        gen_time = time.time() - gen_start
        
        print(f"{response}")
        print(f"⏱️ 生成耗时: {gen_time:.2f} 秒")
        print("-" * 50)
    
    print("\n✅ 所有测试完成！")
    
    # 交互模式
    print("\n💬 进入交互模式 (输入 'exit' 退出):")
    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 再见！")
                break
            
            messages = [{"role": "user", "content": user_input}]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            print("🤖:", end=" ")
            response = generate(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=512,
                verbose=False
            )
            print(response)
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break

if __name__ == "__main__":
    main()
