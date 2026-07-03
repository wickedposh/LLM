import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)

lora_config = LoraConfig(
    r=16,                                   # a bit more capacity: proof structure is richer than Q&A
    lora_alpha=32,                          # 2*r
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_example(row):
    messages = [
        {"role": "user", "content": row["theorem"]},
        {"role": "assistant", "content": row["proof"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = load_dataset("json", data_files="algebra.jsonl", split="train").map(format_example)
print(f"Training examples: {len(dataset)}")

train_config = SFTConfig(
    output_dir="qwen-algebra-lora",
    num_train_epochs=10,               # 20 rows -> a few more passes
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)
trainer = SFTTrainer(model=model, train_dataset=dataset, args=train_config)
trainer.train()
trainer.save_model("qwen-algebra-lora")

# ============================================================
# EVAL: naked / fine-tuned / fine-tuned + context
# ============================================================
ADAPTER_DIR = "qwen-algebra-lora"
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
ft_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
finetuned = PeftModel.from_pretrained(ft_base, ADAPTER_DIR).to(device)

# Context carries the DEFINITIONS the proof questions below actually need (matched!).
CONTEXT = """You are proving linear algebra theorems. Use these definitions:
- A subset W of R^n is a subspace iff: (i) 0 is in W, (ii) W is closed under addition, (iii) W is closed under scalar multiplication.
- The row space of a matrix is the span of its rows.
- A map T is linear iff T(x+y)=T(x)+T(y) and T(cx)=cT(x).
- A square matrix M is invertible iff there is a matrix N with MN=NM=I; to disprove invertibility it suffices to show no N satisfies MN=I.
Write proofs in the structure: ASSUMPTION / GOAL / STEP 1, 2, ... / CONCLUSION."""

def ask(m, question, context=None):
    msgs = ([{"role": "system", "content": context}] if context else []) + \
           [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=200, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).split(question)[-1].strip()

# HELD-OUT proof tasks (NOT in the training file -> tests generalization, not memorization)
questions = [
    "Prove that the row space of a matrix is a subspace.",
    "Prove that if T is a linear map then T(2x) = 2T(x).",
    "Prove that a matrix with two equal rows is not invertible.",
]

for q in questions:
    print("\n" + "=" * 74)
    print("Q:", q)
    print("-" * 74)
    print("1. NAKED                :", ask(base, q))
    print("-" * 74)
    print("2. BASE + CONTEXT       :", ask(base, q, context=CONTEXT))         # no fine-tune!
    print("-" * 74)
    print("3. FINE-TUNED           :", ask(finetuned, q))
    print("-" * 74)
    print("4. FINE-TUNED + CONTEXT :", ask(finetuned, q, context=CONTEXT))
print("\n" + "=" * 74)