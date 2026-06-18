## A from-scratch LLM Stack

from a simple neural network to the Gordon-Ramsey-ish chatbot!

## Components
  - GPT: I built a transformer for GPT with Pytorch and built a Ramsey-ish context generator from scratch (following Karpathy's tutorial).
  - BPE tokenizer: 
  - Chatbot: Instead of simple context generator, I changed it into a communicable chatbot in the use of GPT as an answer generator.
  - Router: I combined a boW model with pure GPT-like generation, so that in certain cases, cosine-similarity confidence gate matches to canned replies while novel inputs fall through to the GPT generation.
    

