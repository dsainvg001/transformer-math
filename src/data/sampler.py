import os
import json
import threading
import queue
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator, Callable
from src.tokenizer.tokenizer import Tokenizer
from src.data.generator import ExpressionGenerator

class ExpressionSampler:
    """
    Manages mathematical dataset generation, streaming, offline dumping, and validation caching.
    Supports:
      1. Streaming training with optional background thread prefetching.
      2. Offline sharded bulk dumps (JSONL format) and streaming shard loader.
      3. Seeding/caching evaluation sets with string-level deduplication.
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_depth: int = 3,
        float_precision: int = 1,
        context_len: int = 64,
        seed: int = 42,
        enabled_ops: Optional[List[str]] = None,
        val_size: int = 500,
        test_size: int = 500
    ):
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.float_precision = float_precision
        self.context_len = context_len
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Instantiate ExpressionGenerator
        self.generator = ExpressionGenerator(
            seed=seed,
            max_depth=max_depth,
            float_precision=float_precision,
            enabled_ops=enabled_ops
        )
        
        # Determine all enabled operations
        if enabled_ops is None:
            self.enabled_ops = list(self.generator.op_registry.keys())
        else:
            self.enabled_ops = [op for op in enabled_ops if op in self.generator.op_registry]
            
        # All categories as combination of op x depth
        self.categories = []
        for op in self.enabled_ops:
            for d in range(1, max_depth + 1):
                self.categories.append(f"{op}_d{d}")
                
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        self.idx_to_category = {i: cat for i, cat in enumerate(self.categories)}
        
        # Threading lock for thread-safe random generator access across background workers
        self._lock = threading.Lock()
        
        # Set of held-out expressions to guarantee no overlap
        self.held_out_exprs = set()
        self.seen_train_exprs = set()
        
        # Pre-generate validation and test splits
        self.val_set = []
        self.test_set = []
        self._generate_val_test_sets(val_size, test_size)

    def _generate_val_test_sets(self, val_size: int, test_size: int):
        """
        Generates fixed, seeded, held-out validation and test sets.
        Ensures uniform distribution over category-difficulty buckets.
        """
        # Save current rng state to avoid interfering with training streams
        state = self.rng.bit_generator.state
        self.rng = np.random.default_rng(self.seed + 9999)
        
        total_needed = val_size + test_size
        generated = 0
        
        # Sample evenly across all enabled categories
        cat_cycle = list(self.categories)
        idx_cycle = 0
        
        temp_list = []
        seen_exprs = set()
        
        # Generate with robust retry limit
        max_tries = total_needed * 10
        tries = 0
        
        while generated < total_needed and tries < max_tries:
            tries += 1
            cat = cat_cycle[idx_cycle % len(cat_cycle)]
            idx_cycle += 1
            
            op, d_str = cat.split("_d")
            depth = int(d_str)
            
            try:
                expr_str, val = self.generator.generate_for_op(op, depth)
                val_str = self.generator.format_result(val)
                if expr_str not in seen_exprs:
                    seen_exprs.add(expr_str)
                    temp_list.append({
                        "expr": expr_str,
                        "val": val_str,
                        "category": cat
                    })
                    generated += 1
            except Exception:
                continue
                
        # Split into validation and test sets
        self.val_set = temp_list[:val_size]
        self.test_set = temp_list[val_size:val_size + test_size]
        
        # Update overall held_out set so training stream dedupes against them
        for item in temp_list:
            self.held_out_exprs.add(item["expr"])
            
        # Restore rng state
        self.rng.bit_generator.state = state

    def tokenize_and_pad(self, expr_str: str, val_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Formats and tokenizes: BOS expr SEP val EOS PAD...
        Returns input_ids (context_len) and loss_mask (context_len - 1)
        """
        expr_ids = self.tokenizer.encode(expr_str, add_bos=True, add_eos=False)
        val_ids = self.tokenizer.encode(val_str, add_bos=False, add_eos=True)
        
        # Total sequence: BOS expr SEP val EOS
        full_ids = expr_ids + [self.tokenizer.sep_id] + val_ids
        
        if len(full_ids) > self.context_len:
            # Truncate if exceeds
            full_ids = full_ids[:self.context_len]
            
        # Pad to context_len
        pad_len = self.context_len - len(full_ids)
        input_ids = full_ids + [self.tokenizer.pad_id] * pad_len
        input_ids_arr = np.array(input_ids, dtype=np.int32)
        
        # Calculate loss mask for target tokens
        # Target sequence is shifted by 1 relative to input during transformer training.
        # So at position i (input_ids[i]), we predict input_ids[i+1].
        # We want to enable loss only when input_ids[i+1] is a result token (including EOS).
        # Which is from after SEP token up to EOS token.
        loss_mask = np.zeros(self.context_len - 1, dtype=np.float32)
        
        try:
            sep_idx = full_ids.index(self.tokenizer.sep_id)
            # Result tokens are from sep_idx+1 to end of actual seq
            # So in y = input_ids[1:], the result starts at sep_idx (since y[sep_idx] = input_ids[sep_idx+1])
            # The length of result is (len(full_ids) - 1) - sep_idx
            result_len = len(full_ids) - 1 - sep_idx
            if result_len > 0:
                loss_mask[sep_idx : sep_idx + result_len] = 1.0
        except ValueError:
            pass # No SEP token found due to extreme truncation
            
        return input_ids_arr, loss_mask

    def generate_single_sample(self, category_probs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generates a single non-degenerate training sample.
        Deduplicates against held-out validation and test sets.
        Thread-safe under multi-worker background prefetching.
        """
        if category_probs is None:
            category_probs = np.ones(len(self.categories)) / len(self.categories)
            
        for _ in range(100):
            with self._lock:
                cat_idx = self.rng.choice(len(self.categories), p=category_probs)
                cat = self.categories[cat_idx]
                op, d_str = cat.split("_d")
                depth = int(d_str)
                
                try:
                    expr_str, val = self.generator.generate_for_op(op, depth)
                    if expr_str in self.held_out_exprs:
                        continue
                    self.seen_train_exprs.add(expr_str)
                    val_str = self.generator.format_result(val)
                except Exception:
                    continue

            input_ids, loss_mask = self.tokenize_and_pad(expr_str, val_str)
            return {
                "expr": expr_str,
                "val": val_str,
                "category": cat,
                "category_idx": cat_idx,
                "input_ids": input_ids,
                "loss_mask": loss_mask
            }
        raise RuntimeError("Failed to generate valid sample from stream.")

    def stream_batches(
        self,
        batch_size: int,
        get_probs_fn: Optional[Callable[[], np.ndarray]] = None,
        prefetch_size: int = 10
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Yields batches of samples. If prefetch_size > 0, does prefetching in a background thread.
        """
        if prefetch_size > 0:
            yield from self._stream_batches_prefetch(batch_size, get_probs_fn, prefetch_size)
        else:
            while True:
                probs = get_probs_fn() if get_probs_fn else None
                yield self._collate_batch(batch_size, probs)

    def _collate_batch(self, batch_size: int, probs: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        input_ids_batch = np.empty((batch_size, self.context_len), dtype=np.int32)
        loss_mask_batch = np.empty((batch_size, self.context_len - 1), dtype=np.float32)
        cat_idx_batch = np.empty(batch_size, dtype=np.int32)
        
        for i in range(batch_size):
            sample = self.generate_single_sample(probs)
            input_ids_batch[i] = sample["input_ids"]
            loss_mask_batch[i] = sample["loss_mask"]
            cat_idx_batch[i] = sample["category_idx"]
            
        return {
            "input_ids": input_ids_batch,
            "loss_mask": loss_mask_batch,
            "category_idx": cat_idx_batch
        }

    def _stream_batches_prefetch(
        self,
        batch_size: int,
        get_probs_fn: Optional[Callable[[], np.ndarray]],
        prefetch_size: int
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        batch_queue = queue.Queue(maxsize=prefetch_size)
        stop_event = threading.Event()

        def producer():
            while not stop_event.is_set():
                probs = get_probs_fn() if get_probs_fn else None
                try:
                    batch = self._collate_batch(batch_size, probs)
                    # Block with timeout to check stop_event regularly
                    while not stop_event.is_set():
                        try:
                            batch_queue.put(batch, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                except Exception as e:
                    import traceback
                    print(f"Warning in prefetch worker thread: {e}", flush=True)
                    traceback.print_exc()
                    continue

        threads = []
        num_workers = min(4, os.cpu_count() or 2)
        for _ in range(num_workers):
            t = threading.Thread(target=producer, daemon=True)
            t.start()
            threads.append(t)

        try:
            while True:
                # Use timeout so we don't hang forever if all worker threads die
                while True:
                    try:
                        batch = batch_queue.get(timeout=30)
                        batch_queue.task_done()
                        yield batch
                        break
                    except queue.Empty:
                        # Check if all workers have died; if so raise to surface the hang
                        alive = any(t.is_alive() for t in threads)
                        if not alive:
                            raise RuntimeError(
                                "All prefetch worker threads have died. "
                                "Check generate_single_sample for errors."
                            )
                        continue
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=0.5)

    # -------------------------------------------------------------
    # Offline Dump and Streaming Loader
    # -------------------------------------------------------------
    def dump_offline_dataset(self, output_dir: str, num_examples: int, shard_size: int = 100000):
        """
        Dumps a large fixed dataset into sharded JSONL files.
        Deduplicates strings to keep the offline dataset clean.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        shard_idx = 0
        current_examples = 0
        seen_exprs = set()
        
        while current_examples < num_examples:
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.jsonl")
            examples_this_shard = min(shard_size, num_examples - current_examples)
            
            with open(shard_path, "w") as f:
                for _ in range(examples_this_shard):
                    # Use flat distribution for offline data dump
                    sample = self.generate_single_sample(None)
                    # Deduplication with bounded retries to avoid infinite loop
                    max_dedup_retries = 50
                    retries = 0
                    while sample["expr"] in seen_exprs and retries < max_dedup_retries:
                        sample = self.generate_single_sample(None)
                        retries += 1
                    seen_exprs.add(sample["expr"])
                    
                    data_item = {
                        "expr": sample["expr"],
                        "val": sample["val"],
                        "category": sample["category"]
                    }
                    f.write(json.dumps(data_item) + "\n")
                    current_examples += 1
                    
            shard_idx += 1

    def stream_offline_dataset(
        self,
        dataset_dir: str,
        batch_size: int,
        shuffle_shards: bool = True
    ) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Streams batches from sharded JSONL files on disk without loading everything into memory.
        """
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Offline dataset directory {dataset_dir} does not exist.")
            
        shards = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jsonl")])
        if len(shards) == 0:
            raise ValueError(f"No JSONL shards found in {dataset_dir}")
            
        while True:
            shards_to_read = list(shards)
            if shuffle_shards:
                self.rng.shuffle(shards_to_read)
                
            input_ids_batch = []
            loss_mask_batch = []
            cat_idx_batch = []
            
            for shard in shards_to_read:
                with open(shard, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        expr_str = item["expr"]
                        val_str = item["val"]
                        cat = item["category"]
                        cat_idx = self.category_to_idx.get(cat, 0)
                        
                        input_ids, loss_mask = self.tokenize_and_pad(expr_str, val_str)
                        input_ids_batch.append(input_ids)
                        loss_mask_batch.append(loss_mask)
                        cat_idx_batch.append(cat_idx)
                        
                        if len(input_ids_batch) == batch_size:
                            yield {
                                "input_ids": np.stack(input_ids_batch, axis=0),
                                "loss_mask": np.stack(loss_mask_batch, axis=0),
                                "category_idx": np.array(cat_idx_batch, dtype=np.int32)
                            }
                            input_ids_batch = []
                            loss_mask_batch = []
                            cat_idx_batch = []
                            
            # Yield remainder if any
            if len(input_ids_batch) > 0:
                yield {
                    "input_ids": np.stack(input_ids_batch, axis=0),
                    "loss_mask": np.stack(loss_mask_batch, axis=0),
                    "category_idx": np.array(cat_idx_batch, dtype=np.int32)
                }
