import math
import numpy as np
import sympy as sp
from typing import Dict, Any, Callable, Tuple, List, Union

class ExpressionGenerator:
    """
    Random mathematical expression generator.
    Produces syntactically and semantically valid math expressions
    for a wide range of operations, including arithmetic and transcendental functions.
    Ensures non-degenerate distributions with controllable depth and operand domains.
    """
    def __init__(self, seed: int = 42, max_depth: int = 3, float_precision: int = 1):
        self.rng = np.random.default_rng(seed)
        self.max_depth = max_depth
        self.float_precision = float_precision
        
        # Registry for operations
        self.op_registry = {}
        self._setup_default_registry()

    def _setup_default_registry(self):
        # Register Arithmetic
        self.register_op("+", arity=2, sympy_fn=lambda x, y: x + y, domain_fn=self._domain_add_sub)
        self.register_op("-", arity=2, sympy_fn=lambda x, y: x - y, domain_fn=self._domain_add_sub)
        self.register_op("*", arity=2, sympy_fn=lambda x, y: x * y, domain_fn=self._domain_mul)
        self.register_op("/", arity=2, sympy_fn=lambda x, y: x / y, domain_fn=self._domain_div)
        self.register_op("^", arity=2, sympy_fn=lambda x, y: x ** y, domain_fn=self._domain_pow)
        
        # Register Transcendental / Elementary Functions
        self.register_op("sin", arity=1, sympy_fn=sp.sin, domain_fn=self._domain_trig)
        self.register_op("cos", arity=1, sympy_fn=sp.cos, domain_fn=self._domain_trig)
        self.register_op("tan", arity=1, sympy_fn=sp.tan, domain_fn=self._domain_trig)
        self.register_op("log", arity=1, sympy_fn=sp.log, domain_fn=self._domain_log)
        self.register_op("ln", arity=1, sympy_fn=sp.log, domain_fn=self._domain_log)
        self.register_op("exp", arity=1, sympy_fn=sp.exp, domain_fn=self._domain_exp)
        self.register_op("sqrt", arity=1, sympy_fn=sp.sqrt, domain_fn=self._domain_sqrt)
        self.register_op("abs", arity=1, sympy_fn=sp.Abs, domain_fn=self._domain_abs)

    def register_op(self, name: str, arity: int, sympy_fn: Callable, domain_fn: Callable):
        """
        Allows registering new operations (one-line add).
        """
        self.op_registry[name] = {
            "arity": arity,
            "sympy_fn": sympy_fn,
            "domain_fn": domain_fn
        }

    # Domain samplers for leaf nodes (numbers)
    def _sample_number(self, num_type: str = "both", low: float = -10.0, high: float = 10.0) -> Union[int, float]:
        if num_type == "both":
            num_type = self.rng.choice(["int", "float"])
        
        if num_type == "int":
            return int(self.rng.integers(int(low), int(high) + 1))
        else:
            val = self.rng.uniform(low, high)
            return round(val, self.float_precision)

    def _domain_add_sub(self) -> Tuple[Union[int, float], Union[int, float]]:
        x = self._sample_number("both", -50.0, 50.0)
        y = self._sample_number("both", -50.0, 50.0)
        return x, y

    def _domain_mul(self) -> Tuple[Union[int, float], Union[int, float]]:
        x = self._sample_number("both", -10.0, 10.0)
        y = self._sample_number("both", -10.0, 10.0)
        return x, y

    def _domain_div(self) -> Tuple[Union[int, float], Union[int, float]]:
        x = self._sample_number("both", -20.0, 20.0)
        # Avoid division by zero
        y_type = self.rng.choice(["int", "float"])
        if y_type == "int":
            y = int(self.rng.choice([i for i in range(-10, 11) if i != 0]))
        else:
            y = float(self.rng.choice([
                self._sample_number("float", 0.5, 10.0),
                self._sample_number("float", -10.0, -0.5)
            ]))
        return x, y

    def _domain_pow(self) -> Tuple[Union[int, float], Union[int, float]]:
        # To avoid complex numbers, base must be positive, and exponent small to avoid overflow
        base = self._sample_number("both", 0.1, 5.0)
        exponent = int(self.rng.choice([-2, -1, 1, 2, 3]))
        return base, exponent

    def _domain_trig(self) -> float:
        # Avoid extremely large angles
        return float(self._sample_number("float", -2 * math.pi, 2 * math.pi))

    def _domain_log(self) -> float:
        # Strictly positive domain
        return float(self._sample_number("float", 0.1, 20.0))

    def _domain_sqrt(self) -> float:
        # Non-negative domain
        return float(self._sample_number("float", 0.0, 50.0))

    def _domain_exp(self) -> float:
        # Keep small to avoid overflow/underflow
        return float(self._sample_number("float", -3.0, 3.0))

    def _domain_abs(self) -> Union[int, float]:
        return self._sample_number("both", -100.0, 100.0)

    def _format_number(self, val: Union[int, float]) -> str:
        if isinstance(val, int) or isinstance(val, np.integer):
            return str(int(val))
        else:
            if abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            return f"{val:.{self.float_precision}f}"

    def format_result(self, val: float) -> str:
        """
        Formats final evaluated target value cleanly to the required float precision.
        """
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        else:
            rounded = round(val, self.float_precision)
            if abs(rounded - round(rounded)) < 1e-9:
                return str(int(round(rounded)))
            return f"{rounded:.{self.float_precision}f}"

    def _build_tree(self, op_name: str, depth: int) -> Tuple[str, Any]:
        """
        Builds expression tree recursively.
        """
        op_info = self.op_registry[op_name]
        arity = op_info["arity"]
        sympy_fn = op_info["sympy_fn"]

        if depth == 1:
            domain_fn = op_info["domain_fn"]
            args = domain_fn()
            if arity == 1:
                val = args[0] if isinstance(args, tuple) else args
                val_str = self._format_number(val)
                sym_val = sp.sympify(val)
                expr_str = f"{op_name}({val_str})"
                sym_expr = sympy_fn(sym_val)
                return expr_str, sym_expr
            else:
                val1, val2 = args
                val1_str = self._format_number(val1)
                val2_str = self._format_number(val2)
                
                # Wrap negative numbers in parents
                val1_str_wrapped = f"({val1_str})" if (isinstance(val1, (int, float)) and val1 < 0) else val1_str
                val2_str_wrapped = f"({val2_str})" if (isinstance(val2, (int, float)) and val2 < 0) else val2_str
                
                expr_str = f"{val1_str_wrapped}{op_name}{val2_str_wrapped}"
                sym_expr = sympy_fn(sp.sympify(val1), sp.sympify(val2))
                return expr_str, sym_expr
        else:
            # depth > 1
            if arity == 1:
                child_op = self.rng.choice(list(self.op_registry.keys()))
                child_str, child_sym = self._build_tree(child_op, depth - 1)
                expr_str = f"{op_name}({child_str})"
                sym_expr = sympy_fn(child_sym)
                return expr_str, sym_expr
            else:
                choice = self.rng.choice(["left", "right", "both"])
                if choice == "left":
                    d_left = depth - 1
                    d_right = int(self.rng.choice(list(range(1, depth))))
                elif choice == "right":
                    d_left = int(self.rng.choice(list(range(1, depth))))
                    d_right = depth - 1
                else:
                    d_left = depth - 1
                    d_right = depth - 1
                
                # Generate left child
                if d_left == 1:
                    l_val = self._sample_number("both", -10.0, 10.0)
                    l_str = self._format_number(l_val)
                    l_str_wrapped = f"({l_str})" if (isinstance(l_val, (int, float)) and l_val < 0) else l_str
                    l_sym = sp.sympify(l_val)
                else:
                    left_op = self.rng.choice(list(self.op_registry.keys()))
                    l_str, l_sym = self._build_tree(left_op, d_left)
                    l_str_wrapped = f"({l_str})"
                
                # Generate right child
                if d_right == 1:
                    r_val = self._sample_number("both", -10.0, 10.0)
                    if op_name == "/" and abs(r_val) < 1e-5:
                        r_val = 1.0
                    r_str = self._format_number(r_val)
                    r_str_wrapped = f"({r_str})" if (isinstance(r_val, (int, float)) and r_val < 0) else r_str
                    r_sym = sp.sympify(r_val)
                else:
                    right_op = self.rng.choice(list(self.op_registry.keys()))
                    r_str, r_sym = self._build_tree(right_op, d_right)
                    r_str_wrapped = f"({r_str})"
                
                expr_str = f"{l_str_wrapped}{op_name}{r_str_wrapped}"
                sym_expr = sympy_fn(l_sym, r_sym)
                return expr_str, sym_expr

    def generate_for_op(self, op_name: str, depth: int) -> Tuple[str, float]:
        """
        Generates a valid mathematical expression for the given operator/function and depth.
        Ensures value is within non-degenerate range.
        """
        for _ in range(200):
            try:
                tree_str, sympy_expr = self._build_tree(op_name, depth)
                val_complex = sympy_expr.evalf()
                
                # Verify we don't have complex results (except extremely tiny residual imaginary part)
                if sp.I in val_complex.free_symbols or val_complex.is_real is False:
                    continue
                
                val = float(val_complex)
                if math.isnan(val) or math.isinf(val) or abs(val) > 1e6:
                    continue
                
                # Double-check: some functions return complex type float with 0.j
                if hasattr(val, "imag") and abs(val.imag) > 1e-9:
                    continue
                
                result_str = self.format_result(val)
                # Extra check: parse result_str back to float to make sure it evaluates
                float(result_str)
                
                return tree_str, float(result_str)
            except Exception:
                continue
        raise RuntimeError(f"Failed to generate non-degenerate expression for {op_name} at depth {depth}")
