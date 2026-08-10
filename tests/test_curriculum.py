import pytest
import numpy as np
from src.data.curriculum import CurriculumTracker

def test_curriculum_initialization():
    categories = ["add_d1", "sub_d1", "sin_d1"]
    tracker = CurriculumTracker(categories, enabled=True, floor_prob=0.05)
    
    # Should start with uniform weights
    probs = tracker.get_probabilities()
    assert len(probs) == 3
    assert np.allclose(probs, [1/3, 1/3, 1/3])

def test_curriculum_disabled_fallback():
    categories = ["add_d1", "sub_d1", "sin_d1"]
    tracker = CurriculumTracker(categories, enabled=False)
    
    # Even if we change losses, if disabled, weights remain uniform
    tracker.update_train_loss(0, 0.1)
    tracker.update_val_losses({"add_d1": 0.1, "sub_d1": 5.0})
    tracker.recompute_weights()
    
    probs = tracker.get_probabilities()
    assert np.allclose(probs, [1/3, 1/3, 1/3])

def test_curriculum_weight_shifts_and_floor():
    categories = ["add_d1", "sub_d1", "sin_d1"]
    tracker = CurriculumTracker(
        categories,
        enabled=True,
        floor_prob=0.1,
        temperature=1.0,
        ema_alpha=1.0 # use 1.0 to overwrite loss immediately for deterministic testing
    )
    
    # Update validation losses: sub_d1 is extremely hard, others easy
    # Let's set train losses reasonably to avoid overfitting flags
    tracker.update_train_loss(0, 0.5)
    tracker.update_train_loss(1, 4.0)
    tracker.update_train_loss(2, 0.5)
    
    tracker.update_val_losses({
        "add_d1": 0.5,
        "sub_d1": 5.0,
        "sin_d1": 0.5
    })
    
    tracker.recompute_weights()
    probs = tracker.get_probabilities()
    
    # sub_d1 has much higher loss, so it should have higher probability
    assert probs[1] > probs[0]
    assert probs[1] > probs[2]
    
    # Check that add_d1 and sin_d1 did not drop below floor_prob (0.1)
    assert probs[0] >= 0.099 # float precision tolerance
    assert probs[2] >= 0.099

def test_curriculum_overfitting_detection():
    categories = ["add_d1", "sub_d1"]
    tracker = CurriculumTracker(
        categories,
        enabled=True,
        floor_prob=0.01,
        temperature=1.0,
        overfit_ratio=3.0,
        overfit_train_threshold=0.2,
        overfit_decay=0.1,
        ema_alpha=1.0
    )
    
    # Case 1: normal high loss, not overfitted (train loss is high, val loss high)
    tracker.update_train_loss(0, 1.0)
    tracker.update_val_losses({"add_d1": 3.5})
    tracker.recompute_weights()
    assert not tracker.overfitted_flags[0]
    
    # Case 2: Overfitted! (train loss extremely low < 0.2, val loss is high > 3x train loss)
    tracker.update_train_loss(1, 0.05)
    tracker.update_val_losses({"sub_d1": 3.0})
    tracker.recompute_weights()
    assert tracker.overfitted_flags[1]
    
    # Check that because sub_d1 is overfitted, its weight is down-weighted
    # Even though its raw val loss is high (3.0 vs add_d1's 3.5), sub_d1 should get less probability
    probs = tracker.get_probabilities()
    # add_d1 (not overfitted, val 3.5) vs sub_d1 (overfitted, val 3.0 decay 0.1 -> eff 0.3)
    assert probs[0] > probs[1]
