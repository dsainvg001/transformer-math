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

def _worker_generate_chunk(args_tuple: Tuple[int, List[str], int, int, int, float]) -> List[Dict[str, Any]]:
    """
    Worker function executed in parallel CPU processes.
    Generates an exact chunk of math expressions uniformly distributed over categories.
    Guaranteed high-throughput execution without stalls.
    """
    worker_id, categories, chunk_size, max_depth, float_precision, int_ratio = args_tuple
    
    # Initialize worker-specific seed with pid and timestamp
    seed = (int(time.time() * 1000) + worker_id * 10007 + os.getpid()) % (2**31 - 1)
    generator = ExpressionGenerator(
        seed=seed,
        max_depth=max_depth,
        float_precision=float_precision,
        enabled_ops=ALL_OPS,
        int_ratio=int_ratio
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
    sharding into JSONL files, auto-resuming from Hugging Face, and batch uploading.
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
        skip_upload: bool,
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        int_ratio: float = 0.5,
        shard_offset: int = 0,
        auto_resume: bool = True,
        upload_interval: int = 300
    ):
        self.num_samples = num_samples
        self.shard_size = shard_size
        self.output_dir = os.path.abspath(output_dir)
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.num_workers = max(1, num_workers)
        self.debug_mode = debug_mode
        self.skip_upload = skip_upload
        self.min_depth = min_depth
        self.max_depth = max_depth if max_depth is not None else (3 if not debug_mode else 2)
        self.int_ratio = int_ratio
        self.shard_offset = shard_offset
        self.auto_resume = auto_resume
        self.upload_interval = upload_interval
        self.last_upload_time = time.time()
        self.float_precision = 1
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build category list: ops x depth range
        self.categories = [f"{op}_d{d}" for op in ALL_OPS for d in range(self.min_depth, self.max_depth + 1)]
        
        # Manifest file for tracking uploaded shards across resumes
        self.manifest_path = os.path.join(self.output_dir, "uploaded_shards.txt")
        self.uploaded_shards = self._load_manifest()
        
        # Initialize Hugging Face API if uploading is enabled
        self.api = None
        if not self.skip_upload:
            self._init_huggingface()
            if self.auto_resume:
                remote_offset = self._detect_remote_shards()
                if remote_offset > self.shard_offset:
                    print(f"[HF RESUME] Auto-resuming shard_offset from {self.shard_offset} -> {remote_offset}")
                    self.shard_offset = remote_offset

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

    def _detect_remote_shards(self) -> int:
        """
        Queries Hugging Face repository to find existing remote dataset shards.
        Returns the next available shard index.
        """
        if not self.api:
            return 0
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            shard_indices = []
            for f in files:
                basename = os.path.basename(f)
                if basename.startswith("shard_") and basename.endswith(".jsonl"):
                    try:
                        idx_str = basename.replace("shard_", "").replace(".jsonl", "")
                        shard_indices.append(int(idx_str))
                    except ValueError:
                        pass
            if shard_indices:
                max_remote = max(shard_indices)
                print(f"[HF RESUME] Detected {len(shard_indices)} existing remote shard(s). Max remote shard index: {max_remote}.")
                return max_remote + 1
        except Exception as e:
            print(f"[HF RESUME] Info inspecting remote repo files: {e}")
        return 0

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

# Transformer Math Dataset ({'DEBUG Mode' if self.debug_mode else f'{self.num_samples:,} Samples Sharded'})

High-precision synthetic mathematical expression dataset generated for training sequence-to-sequence math Transformers in JAX/Flax.

## Dataset Structure

- **Total Samples**: {self.num_samples:,}
- **Shard Format**: JSONL sharded files ({self.shard_size:,} samples per shard)
- **Supported Operations**: `+`, `-`, `*`, `/`, `^`, `sin`, `cos`, `tan`, `log`, `ln`, `exp`, `sqrt`, `abs`
- **Expression Depth Range**: Depth {self.min_depth} to {self.max_depth}
- **Integer Operand Ratio**: {int(self.int_ratio * 100)}%

## Data Fields

Each line in the `.jsonl` shard files is a JSON object with the following fields:

- `expr` (`str`): Syntactically valid mathematical expression (e.g. `"sin((3.5))+cos((1.2))"`)
- `val` (`str`): Target evaluated numerical result formatted to precision (e.g. `"0.6"`)
- `category` (`str`): Operation-depth category bucket (e.g. `"sin_d4"`)
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

    def _flush_batch_upload(self, force: bool = False):
        """
        Uploads accumulated local JSONL shards using upload_large_folder
        to handle massive datasets and avoid Hugging Face rate limits (HTTP 429).
        """
        if not self.api or self.skip_upload:
            return
            
        now = time.time()
        if not force and (now - self.last_upload_time < self.upload_interval):
            return

        unuploaded = [
            f for f in os.listdir(self.output_dir)
            if f.startswith("shard_") and f.endswith(".jsonl") and f not in self.uploaded_shards
        ]
        
        if not unuploaded and not force:
            return

        print(f"\n[HF LARGE FOLDER UPLOAD] Syncing {len(unuploaded)} pending shard(s) to {self.repo_id}...", flush=True)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if hasattr(self.api, "upload_large_folder"):
                    self.api.upload_large_folder(
                        repo_id=self.repo_id,
                        folder_path=self.output_dir,
                        repo_type="dataset",
                        allow_patterns="shard_*.jsonl"
                    )
                else:
                    self.api.upload_folder(
                        folder_path=self.output_dir,
                        path_in_repo="data",
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        allow_patterns="shard_*.jsonl",
                        commit_message=f"Batch upload {len(unuploaded)} shard(s) ({time.strftime('%Y-%m-%d %H:%M:%S')})"
                    )
                for shard_name in unuploaded:
                    self._record_uploaded_shard(shard_name)
                self.last_upload_time = time.time()
                print(f"[HF LARGE FOLDER UPLOAD OK] Successfully synced batch to Hugging Face Hub!")
                break
            except Exception as e:
                print(f"[HF LARGE FOLDER UPLOAD WARNING] Attempt {attempt+1}/{max_retries} failed ({e}).")
                if attempt < max_retries - 1:
                    time.sleep(15)

    def generate_and_upload(self):
        total_shards = (self.num_samples + self.shard_size - 1) // self.shard_size
        print("=========================================================================")
        print(f"[START] Math Transformer Dataset Generator ({'DEBUG MODE' if self.debug_mode else 'PRODUCTION MODE'})")
        print("=========================================================================")
        print(f"- Target Samples: {self.num_samples:,}")
        print(f"- Shard Size: {self.shard_size:,} samples/shard ({total_shards} total shards, offset: {self.shard_offset})")
        print(f"- Depth Range: {self.min_depth} to {self.max_depth}")
        print(f"- Integer Operand Ratio: {int(self.int_ratio * 100)}%")
        print(f"- CPU Worker Processes: {self.num_workers}")
        print(f"- Output Directory: {self.output_dir}")
        print(f"- Hugging Face Repo: {self.repo_id}")
        print(f"- Upload Interval: {self.upload_interval}s (5-min batching)")
        print(f"- Auto-Resume: {self.auto_resume}")
        print("=========================================================================\n")
        
        start_time = time.time()
        samples_generated = 0
        
        with mp.Pool(processes=self.num_workers) as pool:
            for shard_idx in range(total_shards):
                actual_shard_idx = self.shard_offset + shard_idx
                shard_name = f"shard_{actual_shard_idx:05d}.jsonl"
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
                    tasks.append((w_id, self.categories, size_for_worker, self.max_depth, self.float_precision, self.int_ratio))
                
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
                
                # Periodic 5-minute batch upload to avoid Hugging Face rate limits
                self._flush_batch_upload(force=False)

        # Flush final remaining shards at the end
        self._flush_batch_upload(force=True)

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
        description="Generate Math Transformer dataset shards and upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--debug", "--debug-mode",
        action="store_true",
        dest="debug",
        help="Run in fast debug mode (generates small sample set for validation)."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Total number of math expression samples to generate."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Target number of total JSONL shard files (e.g. 1500, 500)."
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Number of samples per JSONL shard file (default: 100,000)."
    )
    parser.add_argument(
        "--shard-offset",
        type=int,
        default=0,
        help="Starting index offset for shard filenames (e.g., 1500 produces shard_01500.jsonl)."
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="Auto-detect existing remote dataset shards on Hugging Face and resume from highest shard index."
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_false",
        dest="auto_resume",
        help="Disable auto-detecting remote shards on Hugging Face."
    )
    parser.add_argument(
        "--upload-interval",
        type=int,
        default=300,
        help="Batch upload interval in seconds to avoid Hugging Face rate limits (default: 300s = 5 mins)."
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=1,
        help="Minimum expression tree depth (default: 1)."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum expression tree depth (default: 3 for standard, 6 for deep)."
    )
    parser.add_argument(
        "--int-ratio",
        type=float,
        default=0.5,
        help="Ratio of integer vs float operands (1.0 = int only, 0.0 = float only, default: 0.5)."
    )
    parser.add_argument(
        "--int-focused",
        action="store_true",
        help="Shortcut for --int-ratio 0.8."
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
        default=None,
        help="Hugging Face Dataset repository ID."
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
    
    # Int ratio selection
    int_ratio = 0.8 if args.int_focused else args.int_ratio
    
    # Resolve default sample count and shard size based on debug mode
    if args.debug:
        shard_size = args.shard_size or 5000
        num_shards = args.num_shards or 2
        num_samples = args.num_samples or (shard_size * num_shards)
        default_repo = "durgasai299792458/mathmetics-dataset-debug"
    else:
        shard_size = args.shard_size or 100000
        num_shards = args.num_shards or 3500
        num_samples = args.num_samples or (shard_size * num_shards)
        default_repo = "durgasai299792458/mathmetics-dataset-custom"
        
    repo_id = args.repo_id or default_repo
    hf_token = args.hf_token or os.environ.get("HFTOKEN") or os.environ.get("HF_TOKEN")
    
    manager = DatasetGeneratorManager(
        num_samples=num_samples,
        shard_size=shard_size,
        output_dir=args.output_dir,
        repo_id=repo_id,
        hf_token=hf_token,
        num_workers=args.num_workers,
        debug_mode=args.debug,
        skip_upload=args.skip_upload,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        int_ratio=int_ratio,
        shard_offset=args.shard_offset,
        auto_resume=args.auto_resume,
        upload_interval=args.upload_interval
    )
    
    manager.generate_and_upload()

if __name__ == "__main__":
    main()
