import jax
import jax.numpy as jnp
import optax
import numpy as np
import pytest
from src.tokenizer.tokenizer import Tokenizer
from src.data.sampler import ExpressionSampler
from src.model.transformer import TransformerDecoder
from src.train import CustomTrainState, train_step

def test_single_train_step_no_nan():
    # Setup tiny model
    vocab_size = 30
    context_len = 16
    
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
    
    # Initialize variables
    rng = jax.random.PRNGKey(0)
    init_key, _ = jax.random.split(rng)
    dummy_input = jnp.zeros((1, context_len), dtype=jnp.int32)
    params = model.init(init_key, dummy_input)["params"]
    
    tx = optax.adam(0.001)
    state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Create mock batch
    batch = {
        "input_ids": jnp.ones((2, context_len), dtype=jnp.int32),
        "loss_mask": jnp.ones((2, context_len - 1), dtype=jnp.float32),
        "category_idx": jnp.zeros((2,), dtype=jnp.int32)
    }
    
    # Run step
    new_state, loss_val, seq_losses = train_step(state, batch)
    
    assert not jnp.isnan(loss_val)
    assert not jnp.isinf(loss_val)
    assert seq_losses.shape == (2,)

def test_overfit_tiny_batch():
    # Show that model can overfit a tiny batch (loss -> 0)
    vocab_size = 30
    context_len = 16
    batch_size = 1
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=2,
        num_heads=2,
        emb_dim=16,
        mlp_dim=32,
        pos_emb_type="learned",
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    rng = jax.random.PRNGKey(42)
    init_key, _ = jax.random.split(rng)
    dummy_input = jnp.zeros((batch_size, context_len), dtype=jnp.int32)
    params = model.init(init_key, dummy_input)["params"]
    
    # Large LR to overfit fast
    tx = optax.adam(0.02)
    state = CustomTrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # Hardcoded single training example
    # Input sequence: [1, 4, 18, 5, 3, 9, 2, 0, 0, ...]
    # Which corresponds to [BOS] 0 + 1 [SEP] 5 [EOS] padded
    input_ids = np.zeros(context_len, dtype=np.int32)
    input_ids[:7] = [1, 4, 18, 5, 3, 9, 2] # BOS, 0, +, 1, SEP, 5, EOS
    
    loss_mask = np.zeros(context_len - 1, dtype=np.float32)
    loss_mask[4:6] = 1.0 # only compute loss on predicting '5' and 'EOS'
    
    batch = {
        "input_ids": jnp.array([input_ids], dtype=jnp.int32),
        "loss_mask": jnp.array([loss_mask], dtype=jnp.float32)
    }
    
    # Train for 50 steps
    initial_loss = None
    final_loss = None
    
    for step in range(60):
        state, loss_val, _ = train_step(state, batch)
        loss_val = float(loss_val)
        if step == 0:
            initial_loss = loss_val
        final_loss = loss_val
        
    print(f"Initial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f}")
    assert final_loss < 0.05
    assert final_loss < initial_loss
