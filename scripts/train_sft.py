#!/usr/bin/env python3
"""
Main Training Script for SFT
Orchestrates supervised fine-tuning with comprehensive logging and error handling
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.sft_trainer import SFTTrainer, SFTConfig


def setup_logging(output_dir: str, verbose: bool = False):
    """
    Configure logging for training
    
    Args:
        output_dir: Directory to save logs
        verbose: Enable verbose logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for Large Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (use subset of data)"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = SFTConfig.from_yaml(args.config)
    
    # Setup logging
    logger = setup_logging(config.output_dir, args.verbose)
    logger.info("=" * 80)
    logger.info("Starting Supervised Fine-Tuning")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Training data: {config.train_file}")
    logger.info(f"Validation data: {config.val_file}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Use LoRA: {args.use_lora}")
    logger.info(f"Debug mode: {args.debug}")
    
    try:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = SFTTrainer(config, use_lora=args.use_lora)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config.output_dir}/final")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
