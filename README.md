  # char-gpt → bpe-gpt: a from-scratch transformer + tokenizer

  A decoder-only GPT and a byte-level BPE tokenizer, both implemented from scratch
  in PyTorch (no `nn.Transformer`, no `tiktoken`), trained to mimic Gordon Ramsay's
  speaking style on a ~280KB Kitchen Nightmares corpus.
  

  ## Architecture


  ## Experiment 1 — learning-rate sweep
  ┌────────┬────────────────────────────────┐
  │   lr   │    training loss @ step 400    │
  ├────────┼────────────────────────────────┤
  │ 0.1    │ 5.84 (plateaued — overshoot)   │
  ├────────┼────────────────────────────────┤
  │ 0.01   │ 2.90 (best, still descending)  │
  ├────────┼────────────────────────────────┤
  │ 0.001  │ 3.86 (slow)                    │
  ├────────┼────────────────────────────────┤
  │ 0.0001 │ 5.89 (barely moved — underfit) │
  └────────┴────────────────────────────────┘ 
  — 1e-2 optimal, robust across BOTH char-level and BPE tokenization
  (reflects the optimization regime, not the vocabulary). The conventional 1e-3
  default underfit at this data scale.
  
  ## Experiment 2 — tokenization granularity (char vs BPE)
  Same architecture, same data, only the tokenizer changed:

  **Char-level (vocab ~65):**
  :Gordon: Why ho the fuck an is or spectualer and!
  Learns speaker structure and register, but not spelling — "style without sense."
  
  **Byte-level BPE (vocab ~1256):**
  :Gordon: If you do not turn it around today, two o'clock...
  :Justin: Fucking dangle me like a fucking puppet!
  Recovers word-level coherence and correct spelling (incl. profanity and accented
  French from bilingual episodes). The structure-vs-spelling tradeoff of tokenization.
  
  ## Experiment 3 — overfitting on a small corpus
  Training the BPE model to 3000 steps drives train loss below 1.0 — but this is
  memorization, not generalization: fluent-looking phrases are recalled training
  fragments. On 280KB, the model lacks data to generalize at this capacity.
  
  ## Takeaways
  Demonstrates the full small-LM arc end to end: underfitting (low lr), divergence
  (high lr), the optimization sweet spot, the tokenization granularity tradeoff,
  and memorization on scarce data. Next: GPT-2 fine-tuning (subword pretraining
  fixes both fragmentation and memorization).
