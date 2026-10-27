import torch
from scaling_gordon.model import GPT, GPTConfig

def test_forward_backward_cpu():
    cfg = GPTConfig(vocab_size=256, block_size=8, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 256)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.tok.weight.grad is not None

def test_context_limit():
    cfg = GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    try:
        model(torch.zeros((1, 9), dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("Expected context length error")
