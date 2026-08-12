import pytest
import numpy as np
from src.tokenizer.tokenizer import Tokenizer
from src.data.sampler import ExpressionSampler
from src.eval import verify_dataset_integrity

def test_data_integrity_and_zero_leakage():
    tokenizer = Tokenizer()
    sampler = ExpressionSampler(
        tokenizer=tokenizer,
        max_depth=2,
        float_precision=1,
        context_len=32,
        seed=42,
        val_size=50,
        test_size=50
    )
    
    # Generate 100 training samples
    for _ in range(100):
        sampler.generate_single_sample()
        
    assert len(sampler.seen_train_exprs) == 100
    
    # Perform dataset integrity audit
    report = verify_dataset_integrity(
        test_set=sampler.test_set,
        held_out_exprs=sampler.held_out_exprs,
        seen_train_exprs=sampler.seen_train_exprs
    )
    
    assert report["total_test_samples"] == 50
    assert report["leaked_samples"] == 0
    assert report["leakage_rate_pct"] == 0.0
    assert report["corrupted_samples"] == 0
    assert report["corruption_rate_pct"] == 0.0
    assert report["is_data_clean"] is True
