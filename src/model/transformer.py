import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any

def get_sinusoidal_embeddings(seq_len: int, d_model: int, dtype=jnp.float32):
    """
    Computes standard sinusoidal positional embeddings.
    """
    pos = jnp.arange(seq_len, dtype=dtype)[:, jnp.newaxis]
    i = jnp.arange(d_model, dtype=dtype)[jnp.newaxis, :]
    angle_rates = 1.0 / jnp.power(10000.0, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    
    # apply sin to even indices, cos to odd indices
    sines = jnp.sin(angle_rads[:, 0::2])
    cosines = jnp.cos(angle_rads[:, 1::2])
    
    # Alternate sines and cosines
    # If d_model is odd, we adjust sizes accordingly
    half_d = d_model // 2
    pos_embeddings = jnp.zeros((seq_len, d_model), dtype=dtype)
    pos_embeddings = pos_embeddings.at[:, 0:2*half_d:2].set(sines[:, :half_d])
    pos_embeddings = pos_embeddings.at[:, 1:2*half_d:2].set(cosines[:, :half_d])
    return pos_embeddings

class TransformerBlock(nn.Module):
    """
    Single Pre-LN Transformer Decoder Layer.
    """
    num_heads: int
    qkv_features: int
    mlp_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None):
        # Pre-LN self attention
        norm_x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            broadcast_dropout=False
        )(norm_x, mask=mask)
        x = x + attn_out
        
        # Pre-LN MLP
        norm_x2 = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        mlp_out = nn.Dense(features=self.mlp_dim, dtype=self.dtype, param_dtype=self.param_dtype)(norm_x2)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(features=x.shape[-1], dtype=self.dtype, param_dtype=self.param_dtype)(mlp_out)
        x = x + mlp_out
        return x

class TransformerDecoder(nn.Module):
    """
    A Decoder-only Pre-LN Transformer model for mathematical formula learning.
    Highly configurable and device-agnostic.
    """
    vocab_size: int
    context_len: int
    num_layers: int
    num_heads: int
    emb_dim: int
    mlp_dim: int
    pos_emb_type: str = "learned" # "learned" or "sinusoidal"
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, input_ids):
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        
        # Token Embeddings
        token_embedder = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.emb_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        x = token_embedder(input_ids)
        
        # Positional Embeddings
        if self.pos_emb_type == "learned":
            pos_embedder = nn.Embed(
                num_embeddings=self.context_len,
                features=self.emb_dim,
                embedding_init=nn.initializers.normal(stddev=0.02),
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )
            positions = jnp.arange(seq_len)[jnp.newaxis, :] # (1, seq_len)
            pos_emb = pos_embedder(positions) # (1, seq_len, emb_dim)
            x = x + pos_emb
        elif self.pos_emb_type == "sinusoidal":
            pos_emb = get_sinusoidal_embeddings(seq_len, self.emb_dim, dtype=self.dtype)
            x = x + pos_emb[jnp.newaxis, :, :]
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pos_emb_type}")
            
        # Causal Attention Mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        causal_mask = causal_mask[jnp.newaxis, jnp.newaxis, :, :] # (1, 1, seq_len, seq_len)
        
        # Transformer Layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                qkv_features=self.emb_dim // self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype
            )(x, mask=causal_mask)
            
        # Final Norm
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)(x)
        
        # Output Logits
        logits = nn.Dense(
            features=self.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )(x)
        
        return logits
