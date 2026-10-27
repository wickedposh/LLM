"""Run from repository root: python -m scaling_gordon.train --data path/to/text.txt"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .model import GPT, GPTConfig

def device_for(choice):
    if choice == "auto":
        return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; install a recent PyTorch on Apple Silicon")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return choice

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="UTF-8 training corpus; never commit private text")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=324)
    p.add_argument("--output", default="runs/gordon")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    raw = Path(args.data).read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long)
    if len(data) <= args.block_size + 1:
        p.error("Corpus must contain more than block-size + 1 bytes")
    cut = int(len(data) * 0.9)
    train = data[:cut]
    valid = data[cut:]
    if len(train) <= args.block_size:
        p.error("Training split too small for block size")
    cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    def batch(source):
        ix = torch.randint(0, len(source) - args.block_size, (args.batch_size,))
        x = torch.stack([source[i:i + args.block_size] for i in ix]).to(device)
        y = torch.stack([source[i + 1:i + args.block_size + 1] for i in ix]).to(device)
        return x, y
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if device == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    start = time.perf_counter()
    model.train()
    for step in range(args.steps):
        x, y = batch(train)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, args.steps // 10) == 0:
            print(f"step={step+1} train_loss={loss.item():.4f}", flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - start
    result = dict(device=device, config=asdict(cfg), steps=args.steps,
                  batch_size=args.batch_size, parameters=sum(p.numel() for p in model.parameters()),
                  seconds=seconds, tokens_per_second=args.steps * args.batch_size * args.block_size / seconds,
                  final_train_loss=float(loss.item()))
    if device == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    elif device == "mps":
        result["current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["note"] = "MPS current allocation is not peak memory"
    if len(valid) > args.block_size:
        model.eval()
        with torch.no_grad():
            vx, vy = batch(valid)
            result["sample_validation_loss"] = float(model(vx, vy)[1].item())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "config": asdict(cfg), "step": args.steps}, out / "checkpoint.pt")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
