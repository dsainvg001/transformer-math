import jax
import jax.numpy as jnp
import pytest
from src.model.transformer import TransformerDecoder

def test_transformer_forward_learned():
    vocab_size = 30
    context_len = 16
    batch_size = 2
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=2,
        num_heads=2,
        emb_dim=8,
        mlp_dim=16,
        pos_emb_type="learned",
        dtype=jnp.float32,
        param_dtype=jnp.float32
    )
    
    # Mock inputs
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (batch_size, context_len), 0, vocab_size)
    
    # Initialize variables
    init_rng, apply_rng = jax.random.split(rng)
    params = model.init(init_rng, input_ids)
    
    # Forward pass
    logits = model.apply(params, input_ids)
    
    assert logits.shape == (batch_size, context_len, vocab_size)
    assert logits.dtype == jnp.float32

def test_transformer_forward_sinusoidal_bf16():
    vocab_size = 30
    context_len = 16
    batch_size = 2
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=2,
        num_heads=2,
        emb_dim=8,
        mlp_dim=16,
        pos_emb_type="sinusoidal",
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32
    )
    
    # Mock inputs
    rng = jax.random.PRNGKey(42)
    input_ids = jax.random.randint(rng, (batch_size, context_len), 0, vocab_size)
    
    # Initialize variables
    init_rng, apply_rng = jax.random.split(rng)
    params = model.init(init_rng, input_ids)
    
    # Forward pass
    logits = model.apply(params, input_ids)
    
    assert logits.shape == (batch_size, context_len, vocab_size)
    assert logits.dtype == jnp.bfloat16
