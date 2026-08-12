import os
import json
import pytest
from generate_dataset import extract_ops_used, DatasetGeneratorManager, ALL_OPS

def test_extract_ops_used():
    expr = "sin((3.5))+cos((1.2))"
    ops = extract_ops_used(expr, ALL_OPS)
    assert "sin" in ops
    assert "+" in ops
    assert "cos" in ops
    assert ops.index("sin") < ops.index("+") < ops.index("cos")

def test_dataset_generator_debug_mode(tmp_path):
    out_dir = str(tmp_path / "shards")
    manager = DatasetGeneratorManager(
        num_samples=200,
        shard_size=100,
        output_dir=out_dir,
        repo_id="dsain/test-math",
        hf_token=None,
        num_workers=2,
        debug_mode=True,
        skip_upload=True
    )
    manager.generate_and_upload()
    
    shards = [f for f in os.listdir(out_dir) if f.endswith(".jsonl")]
    assert len(shards) == 2
    
    first_shard = os.path.join(out_dir, shards[0])
    with open(first_shard, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        assert "expr" in first_line
        assert "val" in first_line
        assert "category" in first_line
        assert "ops_used" in first_line
        assert isinstance(first_line["ops_used"], list)
