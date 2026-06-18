## A from-scratch LLM Stack

from a handwritten-transformer to the Gordon-Ramsay-ish chatbot!

## Components
  - GPT: I built a transformer for GPT with PyTorch and built a Ramsay-ish context generator from scratch (following Karpathy's tutorial).
  - BPE tokenizer: A byte-level BPT tokeniser built from scratch - iteratively merging frequent byte pairs to build the vocabulary with encode/decode.
  - Chatbot: Wrapped the raw generator into a conversational chatbot - turn structure, stop conditions, and a context window - using the GPT to generate replies. 
  - Router: I combined a boW model with pure GPT-like generation, so that in certain cases, cosine-similarity confidence gate matches to canned replies while novel inputs fall through to the GPT generation.
    

