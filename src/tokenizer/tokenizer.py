import json
import os
import re
from typing import List, Dict, Union

class Tokenizer:
    """
    Custom tokenizer with a fixed, small vocabulary for mathematical expressions.
    Tokenizer performs digit-by-digit tokenization of numbers to keep the vocabulary tiny
    and force the model to learn number structure.
    """
    def __init__(self, vocab_path: str = None):
        if vocab_path is None:
            # Locate relative to this file
            vocab_path = os.path.join(os.path.dirname(__file__), "vocab.json")
        
        with open(vocab_path, "r") as f:
            self.vocab: Dict[str, int] = json.load(f)
        
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        
        # Build a regex pattern for tokenization.
        # Order matters: we want to match longer function names like "sin", "cos", etc.,
        # first, then decimal numbers, then operators/parentheses/etc.
        # Actually, let's tokenize character-by-character for most digits, but extract multi-char ops explicitly.
        # Operators / functions:
        funcs = ["sin", "cos", "tan", "log", "ln", "exp", "sqrt", "abs"]
        special_tokens = ["PAD", "BOS", "EOS", "SEP"]
        
        # We can construct a combined regex.
        # Sort functions by length descending just in case.
        funcs_sorted = sorted(funcs, key=len, reverse=True)
        
        # Compile a pattern.
        # Since numbers are digit-by-digit, we can treat them individually:
        # digits 0-9, dot '.', minus '-', plus '+', star '*', slash '/', caret '^', parents '(', ')'
        # Let's match any of the functions first, or any single non-whitespace character.
        # This will perfectly separate multi-char functions, and split numbers into individual digits/dots/minus signs.
        func_pattern = "|".join(re.escape(f) for f in funcs_sorted)
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        
        # Pattern: special tokens first, then functions, then any single char (except spaces)
        self.token_regex = re.compile(f"({special_pattern}|{func_pattern}|\\S)")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab["PAD"]

    @property
    def bos_id(self) -> int:
        return self.vocab["BOS"]

    @property
    def eos_id(self) -> int:
        return self.vocab["EOS"]

    @property
    def sep_id(self) -> int:
        return self.vocab["SEP"]

    def tokenize_string(self, text: str) -> List[str]:
        """
        Splits a string into a list of token strings.
        Example: "sin(-3.14)" -> ["sin", "(", "-", "3", ".", "1", "4", ")"]
        """
        # Normalize spaces
        tokens = self.token_regex.findall(text)
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encodes a math expression text into a list of token IDs.
        """
        tokens = self.tokenize_string(text)
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            else:
                # If we hit an unknown character/token, we raise ValueError or skip. Let's raise.
                raise ValueError(f"Token '{t}' from text '{text}' is not in vocabulary.")
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        """
        Decodes a list of token IDs back into a string expression.
        """
        tokens = []
        for i in ids:
            # Handle JAX or numpy array wrappers
            i_val = int(i)
            if i_val not in self.inv_vocab:
                raise ValueError(f"ID {i_val} not in vocabulary.")
            
            token = self.inv_vocab[i_val]
            if skip_special and token in ["PAD", "BOS", "EOS", "SEP"]:
                continue
            tokens.append(token)
        
        # When reconstructing, we want a clean string.
        # Since digit tokens are separate, we can join them.
        # But let's add minimal spaces for readability around functions and multi-digit operations?
        # Standard: just join with nothing, or maybe reconstruct as a readable math formula.
        # Let's do a simple join first. Since there's no space token, simple join is perfect.
        # But we can also insert spaces intelligently between functions/operators to make it look nicer,
        # but simpler is better: join them directly. Let's check how the user wants it.
        # "sin(-3.14)" tokenized then joined should be "sin(-3.14)". This works perfectly with empty string join!
        return "".join(tokens)
