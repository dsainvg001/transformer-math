import pytest
from src.tokenizer.tokenizer import Tokenizer

def test_tokenizer_basic():
    tokenizer = Tokenizer()
    
    assert tokenizer.vocab_size == 30
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.sep_id == 3

    # Test tokenizing a complex string
    expr = "sin(-3.14) + cos(2) * abs(-5)"
    tokens = tokenizer.tokenize_string(expr)
    
    expected = [
        "sin", "(", "-", "3", ".", "1", "4", ")", "+", "cos", "(", "2", ")", "*", "abs", "(", "-", "5", ")"
    ]
    assert tokens == expected

def test_tokenizer_encode_decode_roundtrip():
    tokenizer = Tokenizer()
    expr = "sin(-3.14)+cos(2)*abs(-5.2)"
    
    encoded = tokenizer.encode(expr, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.bos_id
    assert encoded[-1] == tokenizer.eos_id
    
    decoded = tokenizer.decode(encoded, skip_special=True)
    assert decoded == expr

def test_tokenizer_unknown_token():
    tokenizer = Tokenizer()
    with pytest.raises(ValueError):
        tokenizer.encode("sin(x)") # 'x' is not in vocabulary
