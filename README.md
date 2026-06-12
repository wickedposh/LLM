   # char-gpt → bpe-gpt: a from-scratch transformer + tokenizer

  A decoder-only GPT and a byte-level BPE tokenizer, both implemented from scratch
  in PyTorch — **no `nn.Transformer`, no `tiktoken`** — trained to mimic Gordon
  Ramsay's speaking style on a ~280KB Kitchen Nightmares corpus.

  ## Files  
  - `gptbuilding/ramsey_generation.py` — everything: byte-level BPE tokenizer
    (train / encode / decode), the GPT model, and the training + hyperparameter-sweep
    scripts.
  
  ## Architecture
  Token + positional embeddings → 3 pre-LayerNorm blocks (4-head causal
  self-attention + 4× feed-forward) → final LayerNorm → linear LM head.
  Per-head attention scaled by `1/sqrt(head_size)`; dropout on attention weights
  and FFN outputs.
  
  | hyperparameter | value |
  |---|---|
  | n_layer | 3 |
  | n_head | 4 |
  | n_embd | 64 |
  | block_size | 32 |
  | dropout | 0.2 |

  The tokenizer is byte-level BPE: iterated most-frequent-pair merging, with `encode`
  applying merges in learned (lowest-index-first) order and `decode` mapping token
  ids back through the merge vocabulary. Round-trip verified.

  ## Experiment 1 — learning-rate sweep

  BPE model, 500-step sweep (training loss at step 400):
  
  | lr | training loss @ step 400 |
  |---|---|
  | 0.1 | 5.84 — plateaued (overshoot) |
  | **0.01** | **2.90 — best, still descending** |
  | 0.001 | 3.86 — slow |
  | 0.0001 | 5.89 — barely moved (underfit) |
  
  The conventional `1e-3` default underfit at this data scale; `1e-2` converged
  fastest. The same `1e-2` optimum held on the character-level model too, suggesting
  it reflects the optimization regime rather than the vocabulary.
  
  ## Experiment 2 — tokenization granularity (char vs BPE)
  
  Same architecture, same data — only the tokenizer changed.

  **Character-level (vocab ~65):**
  :Gordon: Why ho the fuck an is or spectualer and!
  :Gordon: Oh, you cooke arre off.
  Learns speaker structure and register, but not spelling — "style without sense."

  **Byte-level BPE (vocab ~1256):**
  :Gordon: If you do not turn it around today, two o'clock...
  :Justin: Fucking dangle me like a fucking puppet!
  Recovers word-level coherence and correct spelling — including profanity and the
  accented French from the show's bilingual episodes. This is the structure-vs-
  spelling tradeoff of tokenization granularity, made visible on one corpus.
  
  ## Experiment 3 — overfitting on a small corpus
  
  Training the BPE model to 3000 steps drives training loss below 1.0 — but this is
  **memorization, not generalization**: the fluent-looking phrases are recalled
  training fragments stitched together with garbled seams. At 280KB, the model lacks
  the data to generalize at this capacity, so lower loss here does *not* mean better
  output.
  
  ## Takeaways
  
  A compact end-to-end tour of small-LM behaviour:
  - **underfitting** (learning rate too low) and **divergence** (too high),
  - the **optimization sweet spot** (`1e-2`, robust across tokenizers),
  - the **tokenization granularity tradeoff** (char vs BPE), and
  - **memorization** on a scarce corpus.
  
  Next step: GPT-2 fine-tuning — subword pretraining supplies the language ability
  this 280KB model can't learn from scratch, fixing both the fragmentation and the
  memorization.
