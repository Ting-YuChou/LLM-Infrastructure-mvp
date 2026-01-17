"""
Model Quantization Utilities
Implements INT8/INT4 quantization for inference optimization
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from optimum.bettertransformer import BetterTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """
    Model quantization for inference optimization
    
    Supports:
    - INT8 quantization (8-bit inference)
    - INT4 quantization (4-bit inference)
    - GPTQ quantization
    - Dynamic quantization
    - Better Transformer optimization
    """
    
    def __init__(self, model_path: str):
        """
        Initialize quantizer
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model_path = model_path
        logger.info(f"Initializing quantizer for: {model_path}")
    
    def quantize_int8(
        self,
        output_path: str,
        device_map: str = "auto"
    ) -> None:
        """
        Quantize model to INT8 using bitsandbytes
        
        Reduces memory by ~50% with minimal quality loss
        
        Args:
            output_path: Path to save quantized model
            device_map: Device mapping strategy
        """
        logger.info("Starting INT8 quantization...")
        
        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # Load model with 8-bit quantization
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Save quantized model
        logger.info(f"Saving INT8 quantized model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Log compression stats
        self._log_compression_stats(self.model_path, output_path)
        
        logger.info("✓ INT8 quantization complete")
    
    def quantize_int4(
        self,
        output_path: str,
        device_map: str = "auto",
        use_double_quant: bool = True
    ) -> None:
        """
        Quantize model to INT4 using bitsandbytes
        
        Reduces memory by ~75% with some quality loss
        
        Args:
            output_path: Path to save quantized model
            device_map: Device mapping strategy
            use_double_quant: Use nested quantization for better compression
        """
        logger.info("Starting INT4 quantization...")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type="nf4",  # NormalFloat4
        )
        
        # Load model with 4-bit quantization
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Save quantized model
        logger.info(f"Saving INT4 quantized model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Log compression stats
        self._log_compression_stats(self.model_path, output_path)
        
        logger.info("✓ INT4 quantization complete")
    
    def dynamic_quantization(
        self,
        output_path: str
    ) -> None:
        """
        Apply PyTorch dynamic quantization
        
        Quantizes weights to INT8 at load time
        
        Args:
            output_path: Path to save quantized model
        """
        logger.info("Starting dynamic quantization...")
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
        )
        
        # Apply dynamic quantization
        logger.info("Applying quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Save
        logger.info(f"Saving dynamically quantized model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        quantized_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info("✓ Dynamic quantization complete")
    
    def apply_better_transformer(
        self,
        output_path: str
    ) -> None:
        """
        Apply BetterTransformer optimization
        
        Fuses operations for faster inference
        Not a quantization method but complementary optimization
        
        Args:
            output_path: Path to save optimized model
        """
        logger.info("Applying BetterTransformer optimization...")
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        )
        
        # Convert to BetterTransformer
        logger.info("Converting to BetterTransformer...")
        model = BetterTransformer.transform(model)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Save
        logger.info(f"Saving optimized model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info("✓ BetterTransformer optimization complete")
    
    def _log_compression_stats(self, original_path: str, quantized_path: str):
        """
        Log compression statistics
        
        Args:
            original_path: Original model path
            quantized_path: Quantized model path
        """
        try:
            # Get directory sizes
            original_size = self._get_directory_size(original_path)
            quantized_size = self._get_directory_size(quantized_path)
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
            size_reduction = ((original_size - quantized_size) / original_size * 100) if original_size > 0 else 0
            
            logger.info("=" * 60)
            logger.info("Compression Statistics:")
            logger.info(f"  Original size:  {self._format_size(original_size)}")
            logger.info(f"  Quantized size: {self._format_size(quantized_size)}")
            logger.info(f"  Compression:    {compression_ratio:.2f}x")
            logger.info(f"  Size reduction: {size_reduction:.1f}%")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.warning(f"Could not compute compression stats: {e}")
    
    def _get_directory_size(self, path: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def benchmark_quantized_model(
    model_path: str,
    test_prompts: list,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Benchmark quantized model performance
    
    Args:
        model_path: Path to quantized model
        test_prompts: List of test prompts
        num_iterations: Number of benchmark iterations
        
    Returns:
        Benchmark results
    """
    import time
    
    logger.info(f"Benchmarking model: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(3):
        inputs = tokenizer(test_prompts[0], return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)
    
    # Benchmark
    logger.info(f"Running benchmark ({num_iterations} iterations)...")
    latencies = []
    
    for i in range(num_iterations):
        prompt = test_prompts[i % len(test_prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
    
    # Compute statistics
    import numpy as np
    
    results = {
        'mean_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
    }
    
    # Log results
    logger.info("Benchmark Results:")
    logger.info(f"  Mean latency:   {results['mean_latency']:.3f}s")
    logger.info(f"  Median latency: {results['median_latency']:.3f}s")
    logger.info(f"  P95 latency:    {results['p95_latency']:.3f}s")
    logger.info(f"  P99 latency:    {results['p99_latency']:.3f}s")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Quantization Utility")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--output-path", required=True, help="Output path")
    parser.add_argument(
        "--method",
        choices=["int8", "int4", "dynamic", "better-transformer"],
        default="int8",
        help="Quantization method"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after quantization")
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = ModelQuantizer(args.model_path)
    
    # Apply quantization
    if args.method == "int8":
        quantizer.quantize_int8(args.output_path)
    elif args.method == "int4":
        quantizer.quantize_int4(args.output_path)
    elif args.method == "dynamic":
        quantizer.dynamic_quantization(args.output_path)
    elif args.method == "better-transformer":
        quantizer.apply_better_transformer(args.output_path)
    
    # Benchmark if requested
    if args.benchmark:
        test_prompts = [
            "Explain quantum computing",
            "What is machine learning?",
            "How does photosynthesis work?",
        ]
        benchmark_quantized_model(args.output_path, test_prompts)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
