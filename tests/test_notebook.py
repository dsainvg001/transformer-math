import os
import sys
import json
import pytest
from unittest.mock import MagicMock

def test_notebook_structure_and_execution():
    notebook_path = "train.ipynb"
    assert os.path.exists(notebook_path), "train.ipynb does not exist"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 6
    
    # Mock matplotlib and IPython.display if not installed in headless environment
    try:
        import matplotlib
        import matplotlib.pyplot
    except ImportError:
        sys.modules["matplotlib"] = MagicMock()
        sys.modules["matplotlib.pyplot"] = MagicMock()
        
    try:
        import IPython
        import IPython.display
    except ImportError:
        sys.modules["IPython"] = MagicMock()
        sys.modules["IPython.display"] = MagicMock()
    
    # Extract code cells and compile/execute in a clean dict scope
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 5
    
    # Combine code cells and run with DEBUG_MODE = True
    full_code = ""
    for cell in code_cells:
        lines = "".join(cell["source"])
        # Skip git clone call in test
        lines = lines.replace('subprocess.run(["git", "clone"', '# subprocess.run(["git", "clone"')
        lines = lines.replace('DEBUG_MODE = False', 'DEBUG_MODE = True')
        full_code += lines + "\n\n"
        
    global_scope = {}
    exec(full_code, global_scope)
    
    assert global_scope.get("DEBUG_MODE") is True
    assert os.path.exists(global_scope.get("dataset_save_dir", "."))
