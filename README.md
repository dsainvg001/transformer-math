# Mathematical Operations Learning with JAX and Flax

This repository contains a complete, device-agnostic, CPU-to-GPU-scalable JAX codebase for training a Pre-LN Decoder-Only Transformer to perform mathematical operations (including arithmetic, trigonometry, log, exp, sqrt, and compositions). 

The training pipeline features an **on-the-fly, adaptive curriculum data generator** that updates sampling weights dynamically based on category-level training and validation losses.

---

## 1. Architecture Choice Justification: Decoder-Only Transformer

We chose a **Decoder-Only Transformer** (with causal self-attention) rather than an Encoder-Decoder model for the following reasons:
1. **Simplified Unified Representation:** It encodes both the prompt expression and the target answer in a single continuous sequence, separated by a special `SEP` token.
2. **Standard and Scalable:** Decoder-only architectures (like LLaMA and GPT-style models) are the modern standard for symbolic reasoning. They scale more predictably and are easier to parallelize.
3. **Loss Masking Efficiency:** By applying a binary `loss_mask`, we only calculate gradients and penalize the model for prediction tokens appearing *after* the `SEP` token (i.e., the target result and the `EOS` token). This gives the model the same conditional generation benefits as an encoder-decoder architecture with simpler, unified attention.

---

## 2. Directory Layout

```text
src/
  tokenizer/
    tokenizer.py   # Tokenizer class (digit-by-digit, fixed vocab)
    vocab.json     # Small vocabulary mapping mathematical tokens
  data/
    generator.py   # Safe random mathematical expression generator
    sampler.py     # ExpressionSampler for streaming, bulk dumping, & caching
    curriculum.py  # CurriculumTracker for adaptive task weighting & overfitting detection
  model/
    transformer.py # Pre-LN Decoder-Only Transformer (Learned vs. Sinusoidal)
  train.py         # Main CLI entry point for training
  eval.py          # Greedy generator and validation accuracy / loss calculator
scripts/
  generate_dataset.py  # Sharded offline dataset generator
configs/
  cpu_dev.yaml     # Fast local CPU smoke-test config
  gpu_train.yaml   # Scaled-up configuration for high-performance GPU training
tests/             # Comprehensive unit test suite
```

---

## 3. Setup Instructions

First, install the required packages in CPU-only mode:

```bash
pip install -r requirements.txt
```

*(Note: For GPU training, install JAX with CUDA support instead: `pip install "jax[cuda12]"`)*

---

## 4. Run the CPU Smoke Test End-to-End

To ensure every code path (data generation, forward, backward, checkpoint saving/loading, and evaluation) works perfectly, execute the CPU smoke test:

```bash
PYTHONPATH=. python3 src/train.py --config configs/cpu_dev.yaml --device_profile cpu_dev
```

This tiny model and dataset will train for 120 steps, periodically logging progress, updating the curriculum tracker, evaluating on validation, and saving checkpoints in `./checkpoints_cpu`.

---

## 5. Scaling to Real GPU Training

On a GPU, you can run the exact same entry points with the scaled config without any code changes:

```bash
PYTHONPATH=. python3 src/train.py --config configs/gpu_train.yaml --device_profile gpu_train
```

Features active on GPU:
- **bfloat16 Mixed Precision:** Enabled by default on GPU for high compute speed, keeping master parameters in float32.
- **Sinusoidal Position Embeddings:** Scales up to longer context lengths.
- **Multi-Device Compatibility:** Works seamlessly across single or multi-device backends.

---

## 6. How the Adaptive Curriculum Works

The `CurriculumTracker` manages task-level difficulty dynamically.

### Algorithm Flow
1. **Dynamic Categories:** Each category represents an `(operator, depth)` pair (e.g. `sin_d1`, `+_d2`).
2. **Running Loss Tracking:** It tracks the running train loss and validation loss per category using an Exponential Moving Average (EMA).
3. **Overfitting Detection & Mitigation:** 
   If a category's train loss is low (under `overfit_train_threshold`), but its validation loss is significantly larger than its train loss (e.g., by more than `overfit_ratio`), the category is flagged as **overfitted**. Its effective loss is scaled down by `overfit_decay` (e.g. `0.1`) to prevent wasting training steps memorizing it.
4. **Softmax Sampling Weights:** Effective losses are converted into sampling weights using Softmax:
   $$P_c \propto \exp\left(\frac{L^{eff}_c}{\text{temperature}}\right)$$
5. **Starvation Prevention:** A `floor_prob` (e.g., `0.01`) is enforced so no category is completely starved, allowing the model to periodically recheck previously solved categories.

### Tuning the Curriculum
- To make the sampling more uniform/flat, **increase** the `temperature` or disable the curriculum entirely by setting `curriculum.enabled: false`.
- To focus heavily on hard/high-loss tasks, **decrease** the `temperature` (e.g., `0.5` or `0.2`).
- To adjust how long past performance is remembered, tweak `ema_alpha` (smaller `ema_alpha` retains history longer).

---

## 7. Generating and Inspecting the Offline Bulk Dataset

If you prefer static precomputed training rather than dynamic streaming, you can generate a sharded dataset to disk:

```bash
PYTHONPATH=. python3 scripts/generate_dataset.py --config configs/cpu_dev.yaml --output_dir ./offline_dataset --num_examples 1000 --shard_size 250
```

This creates sharded `.jsonl` files (e.g. `shard_0000.jsonl` to `shard_0003.jsonl`). Each line is a simple JSON object containing:
- `expr`: The generated mathematical expression (e.g., `sin(cos(0.5))`).
- `val`: The evaluated exact result (e.g., `0.9`).
- `category`: The category label (e.g., `sin_d2`).

The dataset loader in `ExpressionSampler.stream_offline_dataset` can stream these shards directly from disk with zero memory footprint.

---

## 8. Running Unit Tests

The codebase includes comprehensive unit tests covering tokenization, expression safety/validity, dataset loaders, single train step NaNs, batch overfitting, and curriculum updates:

```bash
python3 -m pytest tests
```
