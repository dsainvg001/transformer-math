import pytest
import numpy as np
import tempfile
import os
import shutil
import sympy as sp
from src.tokenizer.tokenizer import Tokenizer
from src.data.generator import ExpressionGenerator
from src.data.sampler import ExpressionSampler

def test_expression_generator():
    generator = ExpressionGenerator(seed=42, max_depth=2, float_precision=1)
    
    # Test generation for a few operators
    ops_to_test = ["+", "-", "*", "/", "sin", "cos", "abs", "sqrt"]
    for op in ops_to_test:
        expr, val = generator.generate_for_op(op, depth=1)
        assert isinstance(expr, str)
        assert isinstance(val, (int, float))
        
        # Verify syntactic correctness and evaluation using sympy / python eval
        # Replace caret with python exponentiation in case python's eval is used
        eval_expr = expr.replace("^", "**")
        
        # Evaluate via SymPy
        sym_expr = sp.sympify(eval_expr)
        sym_val = float(sym_expr.evalf())
        
        # Format and compare
        expected_str = generator.format_result(sym_val)
        assert generator.format_result(val) == expected_str

def test_expression_sampler_streaming():
    tokenizer = Tokenizer()
    sampler = ExpressionSampler(
        tokenizer=tokenizer,
        max_depth=2,
        float_precision=1,
        context_len=32,
        seed=100,
        val_size=10,
        test_size=10
    )
    
    # Check held-out val/test sets are created
    assert len(sampler.val_set) == 10
    assert len(sampler.test_set) == 10
    assert len(sampler.held_out_exprs) == 20
    
    # Check that a single generated sample is correct
    sample = sampler.generate_single_sample()
    assert "input_ids" in sample
    assert "loss_mask" in sample
    assert sample["input_ids"].shape == (32,)
    assert sample["loss_mask"].shape == (31,)
    
    # Stream a batch (sync/no prefetch)
    batch_generator = sampler.stream_batches(batch_size=4, prefetch_size=0)
    batch = next(batch_generator)
    assert batch["input_ids"].shape == (4, 32)
    assert batch["loss_mask"].shape == (4, 31)
    assert batch["category_idx"].shape == (4,)

def test_offline_dump_and_stream():
    tokenizer = Tokenizer()
    sampler = ExpressionSampler(
        tokenizer=tokenizer,
        max_depth=2,
        float_precision=1,
        context_len=32,
        seed=123,
        val_size=5,
        test_size=5
    )
    
    # Create temporary directory for bulk dump
    tmpdir = tempfile.mkdtemp()
    try:
        # Dump 10 examples sharded into files of size 5
        sampler.dump_offline_dataset(tmpdir, num_examples=10, shard_size=5)
        
        # Check files are created
        shards = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(shards) == 2
        
        # Read/Stream using offline loader
        offline_generator = sampler.stream_offline_dataset(tmpdir, batch_size=3, shuffle_shards=False)
        batch = next(offline_generator)
        assert batch["input_ids"].shape == (3, 32)
        assert batch["loss_mask"].shape == (3, 31)
        assert batch["category_idx"].shape == (3,)
        
    finally:
        shutil.rmtree(tmpdir)
