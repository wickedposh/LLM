# Scaling Gordon

A research-engineering extension of the Gordon Ramsay nanoGPT learning project.

## Milestone 1: reproducible single-device training

This directory provides a configurable decoder-only Transformer with causal scaled-dot-product attention, byte-level input encoding, AdamW training, checkpointing, and JSON throughput metrics. It does **not** yet implement distributed training.

### Install

From the repository root, create a Python environment and install a current PyTorch release using the [official selector](https://pytorch.org/get-started/locally/). On Apple Silicon, verify:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Train on the M5 GPU

Provide a local UTF-8 text corpus. For the original Gordon Ramsay dataset, point `--data` at your local `ramsey.txt` file (not committed here).

```bash
python -m scaling_gordon.train --data /path/to/ramsey.txt --device mps --steps 200 --output runs/mps
python -m scaling_gordon.train --data /path/to/ramsey.txt --device cpu --steps 200 --output runs/cpu
```

Compare `runs/mps/metrics.json` with `runs/cpu/metrics.json`. Each run writes a checkpoint. For small models CPU can be faster because device overhead dominates. GPU timing is synchronized. CPU and GPU timings include setup-independent training loop and batch transfer, but do not include startup or model initialization.

This baseline uses raw UTF-8 bytes (vocabulary 256), **not** the original learned BPE merges. Results are therefore not directly comparable to the old tokenizer's losses. The original implementation remains on the `chatbot` branch.

### Initialization fix

With PyTorch's default `nn.Embedding` init (N(0, 1)) and weight tying, logits at init have std ≈ √n_embd, and the model starts as a near-perfect "copy the input" predictor. Re-initializing the token embedding with N(0, 0.02²) (`embedding_init_std`) brings the initial loss to the uniform-prediction value ln 256.

Measured at init (n_embd = 256, random inputs and independent random targets):

| | Logit std | Loss | Prediction |
|---|---|---|---|
| Default N(0, 1) | 19.3 | 174 | — |
| N(0, 0.02²) | 0.32 | 5.59 | std 0.02·√256 = 0.32; loss ln 256 + σ²/2 ≈ 5.60 |

Effect on training (MPS, 50 steps): final train loss 5.44 → 3.35 (n_embd 256) and 4.43 → 3.33 (n_embd 512).

### Planned milestones

1. GPU memory profiling and model-size sweeps.
2. Gradient accumulation, mixed precision where supported, and activation checkpointing.
3. Reproducible experiments and tests.
4. CUDA profiling on NVIDIA hardware.
5. DDP and FSDP with multi-GPU benchmarking.

### Known limitations

- Validation is a single sampled batch, not a full evaluation.
- No resume-from-checkpoint yet.
- No learning-rate schedule or gradient clipping.
- MPS metric reports **current** allocated memory, not peak usage.
- Performance and correctness on the user's M5 have not yet been tested.
