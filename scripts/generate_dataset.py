import os
import argparse
import yaml
from src.tokenizer.tokenizer import Tokenizer
from src.data.sampler import ExpressionSampler

def main():
    parser = argparse.ArgumentParser(description="Generate and dump bulk mathematical dataset offline")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save dataset shards")
    parser.add_argument("--num_examples", type=int, default=100000, help="Total number of examples to generate")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of examples per shard/file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    tokenizer = Tokenizer()
    
    max_depth = config["data"].get("max_depth", 3)
    float_precision = config["data"].get("float_precision", 1)
    context_len = config["model"].get("context_len", 64)
    enabled_ops = config["data"].get("enabled_ops", None)
    seed = config.get("seed", 42)
    
    print(f"Initializing ExpressionSampler with seed={seed}, max_depth={max_depth}...")
    sampler = ExpressionSampler(
        tokenizer=tokenizer,
        max_depth=max_depth,
        float_precision=float_precision,
        context_len=context_len,
        seed=seed,
        enabled_ops=enabled_ops,
        val_size=10, # small validation set since we only need train shards
        test_size=10
    )
    
    print(f"Generating {args.num_examples} examples into '{args.output_dir}' (shards of {args.shard_size})...")
    sampler.dump_offline_dataset(
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        shard_size=args.shard_size
    )
    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    main()
