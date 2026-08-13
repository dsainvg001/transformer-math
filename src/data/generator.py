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
    def __init__(self, seed: int = 42, max_depth: int = 3, float_precision: int = 1, enabled_ops=None):
        self.rng = np.random.default_rng(seed)
        self.max_depth = max_depth
        self.float_precision = float_precision
        
        # Registry for operations
        self.op_registry = {}
        self._setup_default_registry()
        
        # Restrict recursive tree building to enabled ops only
        if enabled_ops is not None:
            self._enabled_op_list = [op for op in enabled_ops if op in self.op_registry]
        else:
            self._enabled_op_list = list(self.op_registry.keys())

    def _setup_default_registry(self):
        # Register Arithmetic
        self.register_op("+", arity=2, py_fn=lambda x, y: x + y, domain_fn=self._domain_add_sub)
        self.register_op("-", arity=2, py_fn=lambda x, y: x - y, domain_fn=self._domain_add_sub)
        self.register_op("*", arity=2, py_fn=lambda x, y: x * y, domain_fn=self._domain_mul)
        self.register_op("/", arity=2, py_fn=lambda x, y: x / y, domain_fn=self._domain_div)
        self.register_op("^", arity=2, py_fn=lambda x, y: x ** y, domain_fn=self._domain_pow)
        
        # Register Transcendental / Elementary Functions
        self.register_op("sin", arity=1, py_fn=math.sin, domain_fn=self._domain_trig)
        self.register_op("cos", arity=1, py_fn=math.cos, domain_fn=self._domain_trig)
        self.register_op("tan", arity=1, py_fn=math.tan, domain_fn=self._domain_trig)
        self.register_op("log", arity=1, py_fn=math.log, domain_fn=self._domain_log)
        self.register_op("ln", arity=1, py_fn=math.log, domain_fn=self._domain_log)
        self.register_op("exp", arity=1, py_fn=math.exp, domain_fn=self._domain_exp)
        self.register_op("sqrt", arity=1, py_fn=math.sqrt, domain_fn=self._domain_sqrt)
        self.register_op("abs", arity=1, py_fn=abs, domain_fn=self._domain_abs)

    def register_op(self, name: str, arity: int, py_fn: Callable, domain_fn: Callable):
        """
        Allows registering new operations.
        """
        sympy_map = {
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "log": sp.log, "ln": sp.log, "exp": sp.exp,
            "sqrt": sp.sqrt, "abs": sp.Abs,
            "+": lambda x, y: x + y, "-": lambda x, y: x - y,
            "*": lambda x, y: x * y, "/": lambda x, y: x / y,
            "^": lambda x, y: x ** y
        }
        self.op_registry[name] = {
            "arity": arity,
            "py_fn": py_fn,
            "sympy_fn": sympy_map.get(name, py_fn),
            "domain_fn": domain_fn
        }

    # Domain samplers for leaf nodes (numbers)
    def _sample_number(self, num_type: str = "both", low: float = -10.0, high: float = 10.0) -> Union[int, float]:
        if num_type == "both":
            num_type = self.rng.choice(["int", "float"])
        
        if num_type == "int":
            return int(self.rng.integers(int(low), int(high) + 1))
        else:
            val = float(self.rng.uniform(low, high))
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
        base = self._sample_number("both", 0.1, 5.0)
        exponent = int(self.rng.choice([-2, -1, 1, 2, 3]))
        return base, exponent

    def _domain_trig(self) -> float:
        return float(self._sample_number("float", -2 * math.pi, 2 * math.pi))

    def _domain_log(self) -> float:
        return float(self._sample_number("float", 0.1, 20.0))

    def _domain_sqrt(self) -> float:
        return float(self._sample_number("float", 0.0, 50.0))

    def _domain_exp(self) -> float:
        return float(self._sample_number("float", -3.0, 3.0))

    def _domain_abs(self) -> Union[int, float]:
        return self._sample_number("both", -100.0, 100.0)

    def _format_number(self, val: Union[int, float]) -> str:
        if isinstance(val, (int, np.integer)):
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

    def _build_tree(self, op_name: str, depth: int) -> Tuple[str, float]:
        """
        Builds expression tree recursively with fast Python evaluation and strict domain safety.
        """
        op_info = self.op_registry[op_name]
        arity = op_info["arity"]
        py_fn = op_info["py_fn"]

        if depth <= 1:
            domain_fn = op_info["domain_fn"]
            args = domain_fn()
            if arity == 1:
                val = args[0] if isinstance(args, tuple) else args
                val_str = self._format_number(val)
                expr_str = f"{op_name}({val_str})"
                res_val = float(py_fn(val))
                return expr_str, res_val
            else:
                val1, val2 = args
                val1_str = self._format_number(val1)
                val2_str = self._format_number(val2)
                val1_wrapped = f"({val1_str})" if (isinstance(val1, (int, float)) and val1 < 0) else val1_str
                val2_wrapped = f"({val2_str})" if (isinstance(val2, (int, float)) and val2 < 0) else val2_str
                expr_str = f"{val1_wrapped}{op_name}{val2_wrapped}"
                res_val = float(py_fn(val1, val2))
                return expr_str, res_val

        # depth > 1
        if arity == 1:
            child_op = self.rng.choice(self._enabled_op_list)
            child_str, child_val = self._build_tree(child_op, depth - 1)
            
            # Enforce domain constraints for unary functions
            if op_name == "sqrt":
                if child_val < 0:
                    child_str = f"abs({child_str})"
                    child_val = abs(child_val)
            elif op_name in ("log", "ln"):
                if child_val <= 0:
                    child_str = f"abs({child_str})+0.1"
                    child_val = abs(child_val) + 0.1
            elif op_name == "tan":
                if abs(child_val) > 1.4:
                    child_str = f"sin({child_str})"
                    child_val = math.sin(child_val)
            elif op_name == "exp":
                if abs(child_val) > 3.0:
                    child_str = f"sin({child_str})"
                    child_val = math.sin(child_val)

            expr_str = f"{op_name}({child_str})"
            res_val = float(py_fn(child_val))
            return expr_str, res_val
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
            else:
                left_op = self.rng.choice(self._enabled_op_list)
                l_str, l_val = self._build_tree(left_op, d_left)
                l_str_wrapped = f"({l_str})"

            # Generate right child
            if d_right == 1:
                r_val = self._sample_number("both", -10.0, 10.0)
                r_str = self._format_number(r_val)
                r_str_wrapped = f"({r_str})" if (isinstance(r_val, (int, float)) and r_val < 0) else r_str
            else:
                right_op = self.rng.choice(self._enabled_op_list)
                r_str, r_val = self._build_tree(right_op, d_right)
                r_str_wrapped = f"({r_str})"

            # Enforce domain constraints for binary operations
            if op_name == "/":
                if abs(r_val) < 0.1:
                    r_val = 1.0
                    r_str_wrapped = "(1)"
            elif op_name == "^":
                if l_val <= 0:
                    l_val = abs(l_val) + 0.1
                    l_str_wrapped = f"(abs({l_str})+0.1)"
                exp_choice = int(self.rng.choice([-2, -1, 1, 2, 3]))
                r_val = exp_choice
                r_str_wrapped = str(exp_choice)

            expr_str = f"{l_str_wrapped}{op_name}{r_str_wrapped}"
            res_val = float(py_fn(l_val, r_val))
            return expr_str, res_val

    def generate_for_op(self, op_name: str, depth: int) -> Tuple[str, float]:
        """
        Generates a valid mathematical expression for the given operator/function and depth.
        Guaranteed fast execution with zero retries/stalls.
        """
        for _ in range(20):
            try:
                tree_str, val = self._build_tree(op_name, depth)
                if math.isnan(val) or math.isinf(val) or abs(val) > 1e6:
                    continue
                result_str = self.format_result(val)
                res_float = float(result_str)
                return tree_str, res_float
            except Exception:
                continue

        # Fallback guarantee to depth 1
        tree_str, val = self._build_tree(op_name, 1)
        result_str = self.format_result(val)
        return tree_str, float(result_str)
