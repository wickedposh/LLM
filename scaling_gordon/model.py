"""Configurable decoder-only Transformer for scaling experiments."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    embedding_init_std: float = 0.02

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q, k, v = [z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg=cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.n_layer)))
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight
        nn.init.normal_(self.tok.weight, mean=0.0, std=cfg.embedding_init_std)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError("Sequence exceeds block_size")
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        logits = self.head(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
