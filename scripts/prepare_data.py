#!/usr/bin/env python3
"""
Data Preparation Script
Creates sample datasets for SFT and RLHF training
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict


def create_sft_dataset(output_dir: str, num_examples: int = 100) -> str:
    """
    Create sample SFT (supervised fine-tuning) dataset
    
    Args:
        output_dir: Directory to save dataset
        num_examples: Number of examples to generate
        
    Returns:
        Path to created dataset file
    """
    print(f"Creating SFT dataset with {num_examples} examples...")
    
    # Sample instruction-response pairs
    base_examples = [
        {
            "prompt": "Explain what machine learning is in simple terms.",
            "completion": "Machine learning is a way for computers to learn from examples without being explicitly programmed. It's like teaching a child by showing them many examples until they understand the pattern. The computer analyzes data, finds patterns, and uses those patterns to make decisions or predictions on new data."
        },
        {
            "prompt": "Write a Python function to reverse a string.",
            "completion": "Here's a simple Python function to reverse a string:\n\n```python\ndef reverse_string(s):\n    return s[::-1]\n```\n\nThis uses Python's slice notation with a step of -1, which reverses the string. You can also use `''.join(reversed(s))` for a more explicit approach."
        },
        {
            "prompt": "What are the main differences between supervised and unsupervised learning?",
            "completion": "Supervised learning uses labeled data where the correct answers are provided during training. The model learns to map inputs to known outputs. Examples include classification and regression.\n\nUnsupervised learning works with unlabeled data, finding hidden patterns or structures without explicit answers. Examples include clustering and dimensionality reduction.\n\nThe key difference is that supervised learning has a 'teacher' (labels), while unsupervised learning discovers patterns independently."
        },
        {
            "prompt": "How do I sort a list in Python?",
            "completion": "You can sort a list in Python in two main ways:\n\n1. In-place sorting (modifies original):\n```python\nmy_list = [3, 1, 4, 1, 5]\nmy_list.sort()\nprint(my_list)  # [1, 1, 3, 4, 5]\n```\n\n2. Return new sorted list:\n```python\nmy_list = [3, 1, 4, 1, 5]\nsorted_list = sorted(my_list)\nprint(sorted_list)  # [1, 1, 3, 4, 5]\n```\n\nBoth support `reverse=True` for descending order and `key=` for custom sorting."
        },
        {
            "prompt": "Explain the concept of recursion with an example.",
            "completion": "Recursion is when a function calls itself to solve a problem by breaking it into smaller, similar sub-problems. Here's a classic example - calculating factorial:\n\n```python\ndef factorial(n):\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n```\n\nFor factorial(5):\n- 5 * factorial(4)\n- 5 * 4 * factorial(3)\n- 5 * 4 * 3 * factorial(2)\n- 5 * 4 * 3 * 2 * factorial(1)\n- 5 * 4 * 3 * 2 * 1 = 120\n\nKey parts: base case (stopping condition) and recursive case (function calling itself)."
        },
    ]
    
    # Replicate examples to reach desired count
    dataset = []
    for i in range(num_examples):
        example = base_examples[i % len(base_examples)].copy()
        example["id"] = i
        dataset.append(example)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ SFT dataset saved to: {output_file}")
    print(f"  Total examples: {len(dataset)}")
    
    # Create smaller validation set (10% of training)
    val_size = max(10, num_examples // 10)
    val_dataset = dataset[:val_size]
    val_file = os.path.join(output_dir, "val.json")
    
    with open(val_file, 'w') as f:
        json.dump(val_dataset, f, indent=2)
    
    print(f"✓ Validation dataset saved to: {val_file}")
    print(f"  Total examples: {len(val_dataset)}")
    
    return output_file


def create_preference_dataset(output_dir: str, num_examples: int = 100) -> str:
    """
    Create sample preference dataset for reward model training
    
    Args:
        output_dir: Directory to save dataset
        num_examples: Number of examples to generate
        
    Returns:
        Path to created dataset file
    """
    print(f"\nCreating preference dataset with {num_examples} examples...")
    
    # Sample preference pairs (chosen vs rejected)
    base_examples = [
        {
            "prompt": "How do I learn programming?",
            "chosen": "Start with Python as it's beginner-friendly. Practice daily with small projects, gradually increasing complexity. Use resources like online tutorials, coding platforms (LeetCode, HackerRank), and build real projects. Join programming communities for support and feedback.",
            "rejected": "Just Google it and you'll figure it out eventually. Programming is hard, you might not be cut out for it."
        },
        {
            "prompt": "What's the best way to lose weight?",
            "chosen": "A healthy approach combines balanced nutrition with regular exercise. Focus on whole foods, portion control, and sustainable habits. Aim for 1-2 pounds per week. Consult a healthcare provider for personalized advice.",
            "rejected": "Stop eating. The less you eat, the more weight you'll lose. Crash diets are the fastest way."
        },
        {
            "prompt": "How can I improve my communication skills?",
            "chosen": "Practice active listening, maintain eye contact, and be clear and concise. Read widely to expand vocabulary. Join groups like Toastmasters for public speaking practice. Ask for feedback and reflect on conversations to continuously improve.",
            "rejected": "Just talk more. Volume is what matters. Interrupt people to get your point across faster."
        },
    ]
    
    # Replicate examples
    dataset = []
    for i in range(num_examples):
        example = base_examples[i % len(base_examples)].copy()
        example["id"] = i
        dataset.append(example)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    output_file = os.path.join(output_dir, "train.json")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Preference dataset saved to: {output_file}")
    print(f"  Total examples: {len(dataset)}")
    
    return output_file


def create_rl_prompts(output_dir: str, num_prompts: int = 50) -> str:
    """
    Create sample prompts for RLHF training
    
    Args:
        output_dir: Directory to save prompts
        num_prompts: Number of prompts to generate
        
    Returns:
        Path to created prompts file
    """
    print(f"\nCreating RL prompts with {num_prompts} examples...")
    
    # Sample prompts for RL training
    base_prompts = [
        "Explain the concept of neural networks.",
        "What are the benefits of regular exercise?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is climate change and why is it important?",
        "How do I start a small business?",
        "What are good study habits for students?",
        "Explain quantum computing in simple terms.",
        "What is the importance of biodiversity?",
        "How can I manage stress effectively?",
    ]
    
    # Replicate prompts
    prompts = []
    for i in range(num_prompts):
        prompts.append({
            "id": i,
            "prompt": base_prompts[i % len(base_prompts)]
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompts
    output_file = os.path.join(output_dir, "rl_prompts.json")
    with open(output_file, 'w') as f:
        json.dump(prompts, f, indent=2)
    
    print(f"✓ RL prompts saved to: {output_file}")
    print(f"  Total prompts: {len(prompts)}")
    
    return output_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Prepare datasets for LLM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base output directory for datasets"
    )
    
    parser.add_argument(
        "--sft-examples",
        type=int,
        default=100,
        help="Number of SFT examples to generate"
    )
    
    parser.add_argument(
        "--preference-examples",
        type=int,
        default=100,
        help="Number of preference examples to generate"
    )
    
    parser.add_argument(
        "--rl-prompts",
        type=int,
        default=50,
        help="Number of RL prompts to generate"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Data Preparation for LLM Training")
    print("=" * 80)
    
    # Create datasets
    sft_dir = os.path.join(args.output_dir, "sft")
    pref_dir = os.path.join(args.output_dir, "preferences")
    prompt_dir = os.path.join(args.output_dir, "prompts")
    
    create_sft_dataset(sft_dir, args.sft_examples)
    create_preference_dataset(pref_dir, args.preference_examples)
    create_rl_prompts(prompt_dir, args.rl_prompts)
    
    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)
    print(f"\nDatasets created in: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review the generated datasets")
    print("2. Customize with your own data")
    print("3. Run training: python scripts/train_sft.py --config config/sft_config.yaml")


if __name__ == "__main__":
    main()
