import os
import sys
import argparse
import yaml
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.jax_utils as jutils
import orbax.checkpoint as ocp
from typing import Dict, Any, Tuple, Optional

from src.tokenizer.tokenizer import Tokenizer
from src.data.sampler import ExpressionSampler
from src.data.curriculum import CurriculumTracker
from src.model.transformer import TransformerDecoder
from src.eval import evaluate_on_dataset

# -------------------------------------------------------------
# Robust Checkpoint Manager
# -------------------------------------------------------------
class CheckpointManager:
    """
    Manages saving and loading of model checkpoints with Orbax.
    Backward-compatible and robust across Orbax API shifts.
    """
    def __init__(self, directory: str):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        # Setup checkpointer and manager
        try:
            self.mngr = ocp.CheckpointManager(
                self.directory,
                options=ocp.CheckpointManagerOptions(max_to_keep=3)
            )
        except Exception:
            self.mngr = ocp.CheckpointManager(
                self.directory,
                ocp.StandardCheckpointer(),
                options=ocp.CheckpointManagerOptions(max_to_keep=3)
            )

    def save(self, step: int, state):
        try:
            self.mngr.save(step, args=ocp.args.StandardSave(state))
        except Exception:
            try:
                self.mngr.save(step, state)
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at step {step}: {e}")

    def restore(self, step: int, state):
        try:
            return self.mngr.restore(step, args=ocp.args.StandardRestore(state))
        except Exception:
            try:
                return self.mngr.restore(step, items=state)
            except Exception as e:
                print(f"Warning: Failed to restore checkpoint from step {step}: {e}")
                return state

    def latest_step(self) -> Optional[int]:
        try:
            return self.mngr.latest_step()
        except Exception:
            return None

    def close(self):
        try:
            self.mngr.wait_until_finished()
        except Exception:
            pass
        try:
            self.mngr.close()
        except Exception:
            pass

# -------------------------------------------------------------
# TrainState Definition
# -------------------------------------------------------------
class CustomTrainState(train_state.TrainState):
    # No extra fields needed unless we want to, but standard TrainState holds step, params, tx, opt_state
    pass

# -------------------------------------------------------------
# Core JIT / pmap Loss and Step Functions
# -------------------------------------------------------------
def compute_loss(logits, targets, mask):
    """
    Computes masked cross entropy loss.
    """
    one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot * log_probs, axis=-1) # (batch_size, seq_len - 1)
    
    masked_loss = loss * mask
    total_loss = jnp.sum(masked_loss)
    total_tokens = jnp.sum(mask)
    
    mean_loss = jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)
    return mean_loss, total_loss, total_tokens

@jax.jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
    """
    JIT-compiled single train step for single-device execution.
    """
    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        
        one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_losses = -jnp.sum(one_hot * log_probs, axis=-1)
        
        masked_losses = token_losses * loss_mask
        
        total_loss = jnp.sum(masked_losses)
        total_tokens = jnp.sum(loss_mask)
        mean_loss = jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)
        
        seq_sums = jnp.sum(masked_losses, axis=-1)
        seq_tokens = jnp.sum(loss_mask, axis=-1)
        seq_losses = jnp.where(seq_tokens > 0, seq_sums / seq_tokens, 0.0)
        
        return mean_loss, (seq_losses, total_loss)
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (mean_loss, (seq_losses, total_loss)), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, mean_loss, seq_losses

def make_parallel_train_step():
    """
    Creates a pmap-compiled multi-device train step function.
    Synchronizes gradients across devices via jax.lax.pmean.
    """
    def parallel_step_fn(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]
        
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, inputs)
            
            one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            token_losses = -jnp.sum(one_hot * log_probs, axis=-1)
            
            masked_losses = token_losses * loss_mask
            
            total_loss = jnp.sum(masked_losses)
            total_tokens = jnp.sum(loss_mask)
            mean_loss = jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0)
            
            seq_sums = jnp.sum(masked_losses, axis=-1)
            seq_tokens = jnp.sum(loss_mask, axis=-1)
            seq_losses = jnp.where(seq_tokens > 0, seq_sums / seq_tokens, 0.0)
            
            return mean_loss, (seq_losses, total_loss)
            
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (mean_loss, (seq_losses, total_loss)), grads = grad_fn(state.params)
        
        grads = jax.lax.pmean(grads, axis_name="batch")
        mean_loss = jax.lax.pmean(mean_loss, axis_name="batch")
        seq_losses = jax.lax.all_gather(seq_losses, axis_name="batch")
        
        new_state = state.apply_gradients(grads=grads)
        return new_state, mean_loss, seq_losses

    return jax.pmap(parallel_step_fn, axis_name="batch")

# -------------------------------------------------------------
# Main CLI Training Flow
# -------------------------------------------------------------
def train(config: Dict[str, Any], device_profile: str):
    print(f"=== Starting Training with Device Profile: {device_profile} ===")
    
    # 1. Device detection and logging
    devices = jax.devices()
    num_devices = jax.local_device_count()
    print(f"JAX local device count: {num_devices}")
    print(f"Available JAX devices: {devices}")
    
    # Define seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)
    
    # Precision configuration
    precision = config.get("precision", "float32")
    if precision == "bfloat16":
        compute_dtype = jnp.bfloat16
        print("Mixed precision ENABLED (bfloat16 compute, float32 master params)")
    else:
        compute_dtype = jnp.float32
        print("Mixed precision DISABLED (float32)")
        
    # 2. Tokenizer setup
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # 3. Data Sampler setup
    max_depth = config["data"].get("max_depth", 3)
    float_precision = config["data"].get("float_precision", 1)
    context_len = config["model"].get("context_len", 64)
    enabled_ops = config["data"].get("enabled_ops", None)
    
    sampler = ExpressionSampler(
        tokenizer=tokenizer,
        max_depth=max_depth,
        float_precision=float_precision,
        context_len=context_len,
        seed=seed,
        enabled_ops=enabled_ops,
        val_size=config["data"].get("val_size", 500),
        test_size=config["data"].get("test_size", 500)
    )
    
    # 4. Curriculum Tracker setup
    cur_cfg = config.get("curriculum", {})
    cur_enabled = cur_cfg.get("enabled", True)
    
    cur_tracker = CurriculumTracker(
        categories=sampler.categories,
        enabled=cur_enabled,
        ema_alpha=cur_cfg.get("ema_alpha", 0.2),
        temperature=cur_cfg.get("temperature", 1.0),
        floor_prob=cur_cfg.get("floor_prob", 0.01),
        overfit_ratio=cur_cfg.get("overfit_ratio", 3.0),
        overfit_train_threshold=cur_cfg.get("overfit_train_threshold", 0.3),
        overfit_decay=cur_cfg.get("overfit_decay", 0.1),
        update_interval=cur_cfg.get("update_interval", 100)
    )
    
    # 5. Model Initialization
    model = TransformerDecoder(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        emb_dim=config["model"]["emb_dim"],
        mlp_dim=config["model"]["mlp_dim"],
        pos_emb_type=config["model"].get("pos_emb_type", "learned"),
        dtype=compute_dtype,
        param_dtype=jnp.float32
    )
    
    key = jax.random.PRNGKey(seed)
    init_key, train_key = jax.random.split(key)
    
    # Model dummy inputs for init
    dummy_input = jnp.zeros((1, context_len), dtype=jnp.int32)
    variables = model.init(init_key, dummy_input)
    params = variables["params"]
    
    # 6. Optimizer & Schedule setup
    total_steps = config["training"]["total_steps"]
    warmup_steps = config["training"].get("warmup_steps", 100)
    base_lr = config["training"]["learning_rate"]
    
    warmup_fn = optax.linear_schedule(0.0, base_lr, warmup_steps)
    cosine_fn = optax.cosine_decay_schedule(base_lr, total_steps - warmup_steps)
    schedule_fn = optax.join_schedules([warmup_fn, cosine_fn], [warmup_steps])
    
    tx = optax.chain(
        optax.clip_by_global_norm(config["training"].get("max_grad_norm", 1.0)),
        optax.adamw(
            learning_rate=schedule_fn,
            weight_decay=config["training"].get("weight_decay", 0.01)
        )
    )
    
    # Create TrainState
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # Setup Multi-device pmap or single device
    batch_size = config["training"]["batch_size"]
    if num_devices > 1:
        if batch_size % num_devices != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by device count {num_devices}")
        per_device_batch = batch_size // num_devices
        print(f"Multi-device training ENABLED ({num_devices} devices, per-device batch size {per_device_batch})")
        p_train_step = make_parallel_train_step()
        replicated_state = jutils.replicate(state)
    else:
        per_device_batch = batch_size
        print("Single-device training ENABLED")
    
    # 7. Resume from Checkpoint if exists
    checkpoint_dir = config["training"].get("checkpoint_dir", "./checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        print(f"Resuming training from checkpoint step: {latest_step}")
        state = checkpoint_manager.restore(latest_step, state)
        if num_devices > 1:
            replicated_state = jutils.replicate(state)
        start_step = latest_step + 1
    else:
        print("No checkpoints found. Starting training from scratch.")
        start_step = 1
        
    # 8. Start training stream
    get_probs_fn = lambda: cur_tracker.get_probabilities()
    
    # We use a non-zero prefetch on GPU but tiny or zero on CPU to save memory and avoid hangs during fast shutdown.
    prefetch_size = 5 if device_profile == "gpu_train" else 0
    train_stream = sampler.stream_batches(
        batch_size=batch_size,
        get_probs_fn=get_probs_fn if cur_enabled else None,
        prefetch_size=prefetch_size
    )
    
    print("\n--- Training Loop Starting ---")
    print(f"Total steps: {total_steps} | Batch size: {batch_size} | Start step: {start_step}")
    
    eval_interval = config["training"].get("eval_interval", 100)
    save_interval = config["training"].get("save_interval", 500)
    log_interval = config["training"].get("log_interval", 10)
    
    start_time = time.time()
    
    for step in range(start_step, total_steps + 1):
        # Draw next batch
        batch = next(train_stream)
        
        # Execute train step
        if num_devices > 1:
            batch_reshaped = {
                "input_ids": batch["input_ids"].reshape(num_devices, per_device_batch, -1),
                "loss_mask": batch["loss_mask"].reshape(num_devices, per_device_batch, -1),
                "category_idx": batch["category_idx"].reshape(num_devices, per_device_batch)
            }
            replicated_state, loss_arr, seq_losses_arr = p_train_step(replicated_state, batch_reshaped)
            loss_val = float(loss_arr[0])
            seq_losses_val = np.array(seq_losses_arr[0]).flatten()
            cat_idxs = batch["category_idx"]
        else:
            state, loss_val, seq_losses_val = train_step(state, batch)
            cat_idxs = batch["category_idx"]
        
        # Force evaluation to update loss EMA
        # Update curriculum train loss EMA
        if cur_enabled:
            for cat_idx, seq_loss in zip(cat_idxs, seq_losses_val):
                cur_tracker.update_train_loss(int(cat_idx), float(seq_loss))
                
        # Recompute curriculum weights periodically
        if cur_enabled and step % cur_tracker.update_interval == 0:
            cur_tracker.recompute_weights()
            print(f"[Step {step}] Recomputed curriculum weights.")
            # Log standard deviation of weights to see how focused it is
            probs = cur_tracker.get_probabilities()
            entropy = -np.sum(probs * np.log(probs + 1e-9))
            print(f"  Curriculum Entropy: {entropy:.4f} | Max prob: {np.max(probs):.4f} for category: {cur_tracker.categories[np.argmax(probs)]}")
            
        # Log training metrics
        if step % log_interval == 0 or step == start_step:
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-5)
            print(f"Step {step}/{total_steps} | Loss: {float(loss_val):.4f} | Speed: {steps_per_sec:.2f} steps/sec")
            
        # Periodic Evaluation on validation set
        if step % eval_interval == 0 or step == total_steps:
            print(f"\n--- Running Evaluation at Step {step} ---")
            current_state = jutils.unreplicate(replicated_state) if num_devices > 1 else state
            eval_metrics = evaluate_on_dataset(
                model=model,
                params=current_state.params,
                tokenizer=tokenizer,
                dataset=sampler.val_set,
                context_len=context_len,
                epsilon=config["training"].get("accuracy_epsilon", 0.05)
            )
            print(f"Validation loss: {eval_metrics['overall/loss']:.4f}")
            print(f"Exact-Match accuracy: {eval_metrics['overall/exact_match']*100:.2f}%")
            print(f"Tolerant accuracy (e={config['training'].get('accuracy_epsilon', 0.05)}): {eval_metrics['overall/tolerant_accuracy']*100:.2f}%")
            
            # Feed validation losses to the curriculum tracker
            if cur_enabled:
                val_losses_to_feed = {}
                for cat in sampler.categories:
                    loss_key = f"category_loss/{cat}"
                    if loss_key in eval_metrics:
                        val_losses_to_feed[cat] = eval_metrics[loss_key]
                cur_tracker.update_val_losses(val_losses_to_feed)
                
            print("------------------------------------------\n")
            
        # Periodic Checkpoint saving
        if step % save_interval == 0 or step == total_steps:
            print(f"Saving checkpoint at step {step}...")
            current_state = jutils.unreplicate(replicated_state) if num_devices > 1 else state
            checkpoint_manager.save(step, current_state)
            print("Checkpoint saved successfully.")
            
    # Clean up and wait for asynchronous checkpointing to finish
    checkpoint_manager.close()
    print("=== Training Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX Transformer Training CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--device_profile", type=str, default="cpu_dev", choices=["cpu_dev", "gpu_train"],
                        help="The device profile to run under")
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    train(config, args.device_profile)
