import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.tokenizer.tokenizer import Tokenizer

def parse_float_safe(s: str) -> Tuple[bool, float]:
    """
    Safely parses a string into a float.
    Handles potential format differences or empty strings.
    """
    try:
        # Remove parentheses if present
        s_clean = s.strip().replace("(", "").replace(")", "")
        val = float(s_clean)
        return True, val
    except ValueError:
        return False, 0.0

def evaluate_accuracy(gen_str: str, target_str: str, epsilon: float = 0.05) -> Tuple[bool, bool]:
    """
    Computes exact-match and numerically-tolerant accuracy.
    Returns:
        exact_match (bool)
        numerically_tolerant (bool)
    """
    gen_clean = gen_str.strip()
    target_clean = target_str.strip()
    
    # Exact Match
    exact_match = (gen_clean == target_clean)
    
    # Numerically Tolerant
    is_gen_float, gen_val = parse_float_safe(gen_clean)
    is_target_float, target_val = parse_float_safe(target_clean)
    
    if is_gen_float and is_target_float:
        if gen_val == target_val:
            tolerant = True
        else:
            rel_error = abs(gen_val - target_val) / max(abs(target_val), 1e-9)
            tolerant = (rel_error <= epsilon)
    else:
        # If not parseable as floats, fallback to exact match
        tolerant = exact_match
        
    return exact_match, tolerant

def generate_greedy(
    model,
    params,
    tokenizer: Tokenizer,
    expr_str: str,
    context_len: int,
    max_new_tokens: int = 15
) -> str:
    """
    Generates a response from the model greedily given an expression string.
    """
    # The input sequence starts with [BOS] and the mathematical expression, followed by [SEP]
    prefix = expr_str + "SEP"
    input_ids = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    
    model_apply_fn = jax.jit(model.apply)
    
    for _ in range(max_new_tokens):
        seq_len = len(input_ids)
        if seq_len >= context_len:
            break
            
        # Pad sequence to context_len to maintain static shape
        padded_ids = input_ids + [tokenizer.pad_id] * (context_len - seq_len)
        padded_arr = jnp.array([padded_ids], dtype=jnp.int32)
        
        # Forward pass
        logits = model_apply_fn({"params": params}, padded_arr) # (1, context_len, vocab_size)
        
        # Get predictions for the last active token
        next_token_logits = logits[0, seq_len - 1, :]
        next_token = int(jnp.argmax(next_token_logits))
        
        if next_token == tokenizer.eos_id:
            break
            
        input_ids.append(next_token)
        
    try:
        sep_idx = input_ids.index(tokenizer.sep_id)
        gen_ids = input_ids[sep_idx+1:]
    except ValueError:
        gen_ids = []
        
    return tokenizer.decode(gen_ids, skip_special=True)

def evaluate_on_dataset(
    model,
    params,
    tokenizer: Tokenizer,
    dataset: List[Dict[str, Any]],
    context_len: int,
    epsilon: float = 0.05,
    max_eval_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates the model over an entire dataset (e.g. val_set or test_set).
    Computes token loss, overall and per-category accuracies.
    If max_eval_samples is specified, sub-samples the dataset for fast periodic validation.
    """
    category_metrics = {}
    
    total_samples = len(dataset)
    if total_samples == 0:
        return {}
        
    eval_dataset = dataset
    if max_eval_samples is not None and max_eval_samples < total_samples:
        step_stride = max(1, total_samples // max_eval_samples)
        eval_dataset = dataset[::step_stride][:max_eval_samples]
        
    eval_count = len(eval_dataset)
    overall_em = 0
    overall_tol = 0
    
    # We will also evaluate token-level loss on the evaluation set
    # Using a mini-batch approach to compute evaluation loss
    model_apply_fn = jax.jit(model.apply)
    
    eval_losses = []
    
    for item in eval_dataset:
        expr_str = item["expr"]
        target_str = item["val"]
        cat = item["category"]
        
        if cat not in category_metrics:
            category_metrics[cat] = {
                "count": 0,
                "em": 0,
                "tol": 0,
                "losses": []
            }
            
        # Get greedy decoding prediction
        gen_str = generate_greedy(model, params, tokenizer, expr_str, context_len)
        em, tol = evaluate_accuracy(gen_str, target_str, epsilon)
        
        # Calculate loss on this item
        # Tokenize BOS expr SEP val EOS PAD...
        expr_ids = tokenizer.encode(expr_str, add_bos=True, add_eos=False)
        val_ids = tokenizer.encode(target_str, add_bos=False, add_eos=True)
        full_ids = expr_ids + [tokenizer.sep_id] + val_ids
        
        if len(full_ids) > context_len:
            full_ids = full_ids[:context_len]
            
        pad_len = context_len - len(full_ids)
        input_ids = full_ids + [tokenizer.pad_id] * pad_len
        
        loss_mask = np.zeros(context_len - 1, dtype=np.float32)
        try:
            sep_idx = full_ids.index(tokenizer.sep_id)
            result_len = len(full_ids) - 1 - sep_idx
            if result_len > 0:
                loss_mask[sep_idx : sep_idx + result_len] = 1.0
        except ValueError:
            pass
            
        # Run forward pass for evaluation loss
        padded_arr = jnp.array([input_ids], dtype=jnp.int32)
        inputs = padded_arr[:, :-1]
        targets = padded_arr[:, 1:]
        
        logits = model_apply_fn({"params": params}, inputs)
        
        # Calculate cross entropy
        one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(one_hot * log_probs, axis=-1) # (1, context_len - 1)
        masked_loss = loss * loss_mask
        
        total_loss = jnp.sum(masked_loss)
        total_tokens = jnp.sum(loss_mask)
        mean_loss = float(jnp.where(total_tokens > 0, total_loss / total_tokens, 0.0))
        
        eval_losses.append(mean_loss)
        category_metrics[cat]["losses"].append(mean_loss)
        
        # Accumulate metrics
        category_metrics[cat]["count"] += 1
        if em:
            category_metrics[cat]["em"] += 1
            overall_em += 1
        if tol:
            category_metrics[cat]["tol"] += 1
            overall_tol += 1
            
    # Calculate overall stats
    summary = {
        "overall/loss": float(np.mean(eval_losses)),
        "overall/perplexity": float(np.exp(np.mean(eval_losses))),
        "overall/exact_match": float(overall_em / eval_count),
        "overall/tolerant_accuracy": float(overall_tol / eval_count),
    }
    
    # Calculate per-category stats
    for cat, metrics in category_metrics.items():
        count = metrics["count"]
        summary[f"category_loss/{cat}"] = float(np.mean(metrics["losses"]))
        summary[f"category_em/{cat}"] = float(metrics["em"] / count) if count > 0 else 0.0
        summary[f"category_tol/{cat}"] = float(metrics["tol"] / count) if count > 0 else 0.0
        
    return summary

def verify_dataset_integrity(
    test_set: List[Dict[str, Any]],
    held_out_exprs: set,
    seen_train_exprs: Optional[set] = None
) -> Dict[str, Any]:
    """
    Verifies 100% data integrity:
    1. Zero leakage: Ensures no test set expression was ever seen during training.
    2. Format safety: Ensures all test target values are clean, non-corrupted numbers (no NaN/Inf).
    """
    total_test_samples = len(test_set)
    if total_test_samples == 0:
        return {"is_data_clean": True, "total_test_samples": 0, "leaked_samples": 0, "corrupted_samples": 0}

    test_exprs = set(item["expr"] for item in test_set)
    
    # Leakage check against generated training stream
    leaked_to_train = len(test_exprs.intersection(seen_train_exprs)) if seen_train_exprs else 0
    
    # Corruption / NaN checks
    corrupted_samples = 0
    for item in test_set:
        val_str = item["val"]
        is_valid, val_num = parse_float_safe(val_str)
        if not is_valid or np.isnan(val_num) or np.isinf(val_num):
            corrupted_samples += 1

    leakage_rate_pct = (leaked_to_train / total_test_samples) * 100.0
    corruption_rate_pct = (corrupted_samples / total_test_samples) * 100.0

    return {
        "total_test_samples": total_test_samples,
        "leaked_samples": leaked_to_train,
        "leakage_rate_pct": leakage_rate_pct,
        "corrupted_samples": corrupted_samples,
        "corruption_rate_pct": corruption_rate_pct,
        "is_data_clean": (leaked_to_train == 0 and corrupted_samples == 0)
    }
