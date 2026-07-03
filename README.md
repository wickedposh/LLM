# Fine-Tuning a Small LLM to Produce Structured Linear-Algebra Proofs

An experiment in teaching a small open language model (**Qwen2.5-0.5B-Instruct**) to
generate proofs of basic linear-algebra theorems in a consistent structural format
(`ASSUMPTION → GOAL → STEP 1, 2, … → CONCLUSION`), using **LoRA** parameter-efficient
fine-tuning — and, critically, an **honest evaluation** of what fine-tuning did and did
not achieve.

## What this does

1. **Fine-tunes** Qwen2.5-0.5B-Instruct with a **LoRA adapter** (rank 16, ~0.1% of
   parameters trained) on a hand-built dataset of ~20 structured proofs
   (`algebra.jsonl`), so the model learns the *skeleton* of a proof.
2. **Evaluates** the result on **held-out** theorems (not in the training set) to test
   *generalization*, not memorization.
3. **Ablates** the contribution of fine-tuning vs. prompting by comparing four conditions:
   base model / base + context / fine-tuned / fine-tuned + context.

## Key finding (honest)

- **Fine-tuning successfully transferred proof *structure*** — the model reproduces the
  `ASSUMPTION / GOAL / STEP / CONCLUSION` skeleton on theorems it never saw.
- **It did *not* reliably transfer mathematical *correctness*** — a 0.5B model with ~20
  examples learns the *form* of a proof, not the *reasoning* to fill it. Proofs are
  well-structured but sometimes contain incorrect steps.
- **Ablation result:** a structured *prompt alone* (no fine-tuning) recovers much of the
  organizational benefit; fine-tuning mainly adds *consistent house-style formatting*.
  This is consistent with the general principle that **fine-tuning is best for
  behavior/style and prompting/RAG for supplying facts** — reproduced here on a concrete
  domain.

*Takeaway: structure is learnable from few examples; correctness requires more data, a
larger model, or a verifier. Reported as-is rather than overclaimed.*

## Stack
Python 3.11 · PyTorch (MPS/Apple Silicon) · Hugging Face `transformers`, `peft`, `trl`,
`datasets`. Runs on a laptop (Apple M-series, no GPU cluster required).


