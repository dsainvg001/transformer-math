import os
import pytest

# Ensure 2 CPU devices for testing JAX pmap
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import optax
import numpy as np
import flax.jax_utils as jutils

from src.tokenizer.tokenizer import Tokenizer
from src.model.transformer import TransformerDecoder
from src.train import CustomTrainState, train_step, make_parallel_train_step

def test_multi_gpu_train_step():
    num_devices = jax.local_device_count()
    assert num_devices == 2, f"Expected 2 devices, got {num_devices}"
    
    vocab_size = 30
    context_len = 16
    per_device_batch = 4
    total_batch = per_device_batch * num_devices
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=1,
        num_heads=1,
        emb_dim=8,
        mlp_dim=16,
        pos_emb_type="learned",
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, context_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)["params"]
    
    tx = optax.adam(0.001)
    state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Replicate state for multi-GPU pmap
    replicated_state = jutils.replicate(state)
    
    # Create batch reshaped for pmap: (num_devices, per_device_batch, ...)
    input_ids = np.random.randint(0, vocab_size, (num_devices, per_device_batch, context_len), dtype=np.int32)
    loss_mask = np.ones((num_devices, per_device_batch, context_len - 1), dtype=np.float32)
    
    batch = {
        "input_ids": jnp.array(input_ids),
        "loss_mask": jnp.array(loss_mask)
    }
    
    p_step = make_parallel_train_step()
    new_replicated_state, mean_loss, seq_losses = p_step(replicated_state, batch)
    
    # Single device unreplicate
    unreplicated_state = jutils.unreplicate(new_replicated_state)
    
    assert not jnp.isnan(mean_loss[0])
    assert not jnp.isinf(mean_loss[0])
    assert seq_losses[0].flatten().shape == (total_batch,)
    assert unreplicated_state.step == 1
