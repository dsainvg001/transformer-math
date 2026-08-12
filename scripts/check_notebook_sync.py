"""
check_notebook_sync.py
Validates that train.ipynb and train_tpu.ipynb both contain all required imports.
Run before every git push to catch missing imports early.
"""
import json

REQUIRED_IMPORTS = [
    "import numpy as np",
    "import json",
    "import time",
    "import jax",
    "import jax.numpy as jnp",
    "import optax",
    "import flax.jax_utils as jutils",
    "from src.tokenizer.tokenizer import Tokenizer",
    "from src.data.sampler import ExpressionSampler",
    "from src.data.curriculum import CurriculumTracker",
    "from src.model.transformer import TransformerDecoder",
    "from src.train import make_parallel_train_step",
    "from src.eval import evaluate_on_dataset",
    "RobustCheckpointManager",
    "CheckpointManager",
]

NOTEBOOKS = ["train.ipynb", "train_tpu.ipynb"]

errors = []
for nb_path in NOTEBOOKS:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    full_source = "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    for req in REQUIRED_IMPORTS:
        if req not in full_source:
            errors.append(f"  [{nb_path}] MISSING: {req}")

if errors:
    print("FAIL: Notebook sync check FAILED:")
    for e in errors:
        print(e)
    raise SystemExit(1)
else:
    print("OK: Both notebooks contain all required imports.")
