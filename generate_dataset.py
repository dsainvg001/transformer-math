import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from typing import List, Dict, Any, Tuple, Optional

# Attempt importing huggingface_hub; provide automatic fallback guidance if missing
try:
    from huggingface_hub import HfApi, create_repo
    HAS_HF = True
except ImportError:
    HAS_HF = False

from src.tokenizer.tokenizer import Tokenizer
from src.data.generator import ExpressionGenerator

# List of all standard mathematical operators and functions supported by the generator
ALL_OPS = ["+", "-", "*", "/", "^", "sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"]

def extract_ops_used(expr_str: str, all_ops: List[str] = ALL_OPS) -> List[str]:
    """
    Extracts all mathematical operators and functions present in the expression string.
    Returned in order of appearance in the string.
    """
    found = []
    # Sort ops by length descending so multi-character functions (e.g. 'sin') match before '+'
    sorted_ops = sorted(all_ops, key=len, reverse=True)
    for op in sorted_ops:
        if op in expr_str and op not in found:
            found.append(op)
    # Sort found ops according to their first occurrence index in the expression string
    found.sort(key=lambda op: expr_str.find(op))
    return found

def _worker_generate_chunk(args_tuple: Tuple[int, List[str], int, int, int]) -> List[Dict[str, Any]]:
    """
    Worker function executed in parallel CPU processes.
    Generates an exact chunk of math expressions uniformly distributed over categories.
    Guaranteed high-throughput execution without stalls.
    """
    worker_id, categories, chunk_size, max_depth, float_precision = args_tuple
    
    # Initialize worker-specific seed with pid and timestamp
    seed = (int(time.time() * 1000) + worker_id * 10007 + os.getpid()) % (2**31 - 1)
    generator = ExpressionGenerator(
        seed=seed,
        max_depth=max_depth,
        float_precision=float_precision,
        enabled_ops=ALL_OPS
    )
    
    samples = []
    num_categories = len(categories)
    
    while len(samples) < chunk_size:
        idx = len(samples)
        cat = categories[(worker_id + idx) % num_categories]
        op, d_str = cat.split("_d")
        depth = int(d_str)
        
        try:
            tree_str, val = generator.generate_for_op(op, depth)
            val_str = generator.format_result(val)
            ops_used = extract_ops_used(tree_str, ALL_OPS)
            
            samples.append({
                "expr": tree_str,
                "val": val_str,
                "category": cat,
                "ops_used": ops_used
            })
        except Exception:
            continue
            
    return samples

class DatasetGeneratorManager:
    """
    Manages high-throughput parallel generation of large math transformer datasets,
    sharding into JSONL files, and uploading to Hugging Face Hub.
    """
    def __init__(
        self,
        num_samples: int,
        shard_size: int,
        output_dir: str,
        repo_id: str,
        hf_token: Optional[str],
        num_workers: int,
        debug_mode: bool,
        skip_upload: bool
    ):
        self.num_samples = num_samples
        self.shard_size = shard_size
        self.output_dir = os.path.abspath(output_dir)
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.num_workers = max(1, num_workers)
        self.debug_mode = debug_mode
        self.skip_upload = skip_upload
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build category list: 13 ops x depths = uniform categories
        self.max_depth = 3 if not debug_mode else 2
        self.float_precision = 1
        self.categories = [f"{op}_d{d}" for op in ALL_OPS for d in range(1, self.max_depth + 1)]
        
        # Manifest file for tracking uploaded shards across resumes
        self.manifest_path = os.path.join(self.output_dir, "uploaded_shards.txt")
        self.uploaded_shards = self._load_manifest()
        
        # Initialize Hugging Face API if uploading is enabled
        self.api = None
        if not self.skip_upload:
            self._init_huggingface()

    def _load_manifest(self) -> set:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _record_uploaded_shard(self, shard_name: str):
        self.uploaded_shards.add(shard_name)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(f"{shard_name}\n")

    def _init_huggingface(self):
        if not HAS_HF:
            print("ERROR: huggingface_hub is not installed. Please run: pip install huggingface_hub", file=sys.stderr)
            sys.exit(1)
            
        if not self.hf_token:
            print("ERROR: Hugging Face token missing!", file=sys.stderr)
            print("Please set the HFTOKEN or HF_TOKEN environment variable, or pass --hf-token <TOKEN>", file=sys.stderr)
            sys.exit(1)
            
        print(f"[HF] Authenticating with Hugging Face Hub (Repo: {self.repo_id})...")
        self.api = HfApi(token=self.hf_token)
        try:
            create_repo(
                repo_id=self.repo_id,
                token=self.hf_token,
                repo_type="dataset",
                exist_ok=True,
                private=False
            )
            print(f"[HF] Dataset repository ready: https://huggingface.co/datasets/{self.repo_id}")
            self._upload_dataset_card()
        except Exception as e:
            print(f"Warning during HF repo initialization: {e}")

    def _upload_dataset_card(self):
        """
        Creates and uploads a comprehensive dataset card (README.md) to Hugging Face.
        """
        readme_content = f"""---
license: mit
task_categories:
- text-generation
- mathematical-modeling
language:
- en
tags:
- math
- transformer
- symbolic-math
- jax
- arithmetic
size_categories:
- {"10K-100K" if self.debug_mode else "100M-1B"}
---

# Transformer Math Dataset ({'DEBUG Mode' if self.debug_mode else '250M Production Shards'})

High-precision synthetic mathematical expression dataset generated for training sequence-to-sequence math Transformers in JAX/Flax.

## Dataset Structure

- **Total Samples**: {self.num_samples:,}
- **Shard Format**: JSONL sharded files ({self.shard_size:,} samples per shard)
- **Supported Operations**: `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `tan`, `log`, `ln`, `exp`, `sqrt`, `abs`
- **Max Expression Depth**: {self.max_depth}

## Data Fields

Each line in the `.jsonl` shard files is a JSON object with the following fields:

- `expr` (`str`): Syntactically valid mathematical expression (e.g. `"sin((3.5))+cos((1.2))"`)
- `val` (`str`): Target evaluated numerical result formatted to precision (e.g. `"0.6"`)
- `category` (`str`): Operation-depth category bucket (e.g. `"sin_d2"`)
- `ops_used` (`list[str]`): List of mathematical functions/operators present in the expression (e.g. `["sin", "+", "cos"]`)

## Usage Example

```python
from datasets import load_dataset

dataset = load_dataset("{self.repo_id}", streaming=True)
for sample in dataset["train"]:
    print(sample["expr"], "->", sample["val"], "ops:", sample["ops_used"])
    break
```
"""
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
            
        if self.api and not self.skip_upload:
            try:
                self.api.upload_file(
                    path_or_fileobj=readme_path,
                    path_in_repo="README.md",
                    repo_id=self.repo_id,
                    repo_type="dataset"
                )
                print("[HF] Uploaded Dataset Card (README.md) to Hugging Face.")
            except Exception as e:
                print(f"Warning uploading README.md: {e}")

    def generate_and_upload(self):
        total_shards = (self.num_samples + self.shard_size - 1) // self.shard_size
        print("=========================================================================")
        print(f"[START] Math Transformer Dataset Generator ({'DEBUG MODE' if self.debug_mode else 'PRODUCTION MODE'})")
        print("=========================================================================")
        print(f"- Target Samples: {self.num_samples:,}")
        print(f"- Shard Size: {self.shard_size:,} samples/shard ({total_shards} total shards)")
        print(f"- CPU Worker Processes: {self.num_workers}")
        print(f"- Output Directory: {self.output_dir}")
        print(f"- Hugging Face Repo: {self.repo_id}")
        print(f"- Upload Enabled: {not self.skip_upload}")
        print("=========================================================================\n")
        
        start_time = time.time()
        samples_generated = 0
        
        with mp.Pool(processes=self.num_workers) as pool:
            for shard_idx in range(total_shards):
                shard_name = f"shard_{shard_idx:05d}.jsonl"
                shard_path = os.path.join(self.output_dir, shard_name)
                
                # Check if this shard was already generated and uploaded in a previous run
                if shard_name in self.uploaded_shards and os.path.exists(shard_path):
                    print(f"[SKIP] [Shard {shard_idx+1}/{total_shards}] {shard_name} already uploaded. Skipping...")
                    samples_generated += self.shard_size
                    continue
                    
                items_needed = min(self.shard_size, self.num_samples - samples_generated)
                if items_needed <= 0:
                    break
                    
                chunk_per_worker = items_needed // self.num_workers
                remainder = items_needed % self.num_workers
                
                tasks = []
                for w_id in range(self.num_workers):
                    size_for_worker = chunk_per_worker + (1 if w_id < remainder else 0)
                    tasks.append((w_id, self.categories, size_for_worker, self.max_depth, self.float_precision))
                
                # Run parallel generation across CPU worker processes
                shard_samples = []
                results = pool.map(_worker_generate_chunk, tasks)
                for res in results:
                    shard_samples.extend(res)
                    
                # Write shard to JSONL file
                with open(shard_path, "w", encoding="utf-8") as f:
                    for item in shard_samples:
                        f.write(json.dumps(item) + "\n")
                        
                samples_generated += len(shard_samples)
                elapsed = time.time() - start_time
                rate = samples_generated / max(elapsed, 1e-5)
                eta_sec = (self.num_samples - samples_generated) / max(rate, 1e-5)
                
                print(f"[SHARD] [Shard {shard_idx+1}/{total_shards}] Created {shard_name} ({len(shard_samples):,} samples) | Speed: {rate:,.0f} samples/sec | ETA: {eta_sec/60:.1f} min")
                
                # Upload shard to Hugging Face Hub with retry
                if self.api and not self.skip_upload:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            print(f"[UPLOAD] Uploading {shard_name} to Hugging Face ({self.repo_id})...", flush=True)
                            self.api.upload_file(
                                path_or_fileobj=shard_path,
                                path_in_repo=f"data/{shard_name}",
                                repo_id=self.repo_id,
                                repo_type="dataset"
                            )
                            self._record_uploaded_shard(shard_name)
                            print(f"[OK] Successfully uploaded {shard_name} to Hugging Face.")
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"[RETRY] Upload {shard_name} failed ({e}). Retrying in 2s...")
                                time.sleep(2)
                            else:
                                print(f"[ERROR] Failed to upload {shard_name} after {max_retries} retries: {e}")

        total_elapsed = time.time() - start_time
        print("\n=========================================================================")
        print(f"[COMPLETE] Dataset Generation & Upload Complete!")
        print(f"- Total Samples Generated: {samples_generated:,}")
        print(f"- Total Time Elapsed: {total_elapsed / 60:.2f} minutes")
        print(f"- Average Generation Speed: {samples_generated / max(total_elapsed, 1e-5):,.0f} samples/sec")
        if not self.skip_upload:
            print(f"- Hugging Face Dataset URL: https://huggingface.co/datasets/{self.repo_id}")
        print("=========================================================================")

def main():
    parser = argparse.ArgumentParser(
        description="Generate 250M Math Transformer dataset shards and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--debug", "--debug-mode",
        action="store_true",
        dest="debug",
        help="Run in fast debug mode (generates 10,000 samples for quick validation)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Total number of math expression samples to generate (default: 250,000,000 for prod, 10,000 for debug)."
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of samples per JSONL shard file (default: 100,000 for prod, 5,000 for debug)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hf_dataset_shards",
        help="Local directory to store JSONL dataset shards (default: ./hf_dataset_shards)."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="durgasai299792458/mathmetics-dataset",
        help="Hugging Face Dataset repository ID (default: durgasai299792458/mathmetics-dataset)."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API Token. If not passed, reads from environment variable HFTOKEN or HF_TOKEN."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel CPU processes (default: all {mp.cpu_count()} CPU cores)."
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Generate local JSONL shards without uploading to Hugging Face Hub."
    )
    
    args = parser.parse_args()
    
    # Resolve default sample count and shard size based on debug mode
    if args.debug:
        num_samples = args.num_samples or 10000
        shard_size = args.shard_size or 5000
        repo_id = args.repo_id if args.repo_id != "durgasai299792458/mathmetics-dataset" else "durgasai299792458/mathmetics-dataset-debug"
    else:
        num_samples = args.num_samples or 250000000
        shard_size = args.shard_size or 100000
        repo_id = args.repo_id
        
    # Read token from environment variable HFTOKEN or HF_TOKEN
    hf_token = args.hf_token or os.environ.get("HFTOKEN") or os.environ.get("HF_TOKEN")
    
    manager = DatasetGeneratorManager(
        num_samples=num_samples,
        shard_size=shard_size,
        output_dir=args.output_dir,
        repo_id=repo_id,
        hf_token=hf_token,
        num_workers=args.num_workers,
        debug_mode=args.debug,
        skip_upload=args.skip_upload
    )
    
    manager.generate_and_upload()

if __name__ == "__main__":
    main()
