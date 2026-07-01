"""
Step 6: Interactive chat loop. Loads the trained checkpoint and lets you talk to it.
This mimics the ChatGPT pipeline: prompt -> tokenize -> autoregressive generate -> detokenize.
"""
import torch
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT

TOKENIZER_DIR = "tokenizer"
CKPT_PATH = "checkpoints/mini_llm.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = ByteLevelBPETokenizer(
    f"{TOKENIZER_DIR}/vocab.json", f"{TOKENIZER_DIR}/merges.txt"
)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
cfg = ckpt["config"]
model = MiniGPT(**cfg).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

print("Mini-LLM chat ready. Type 'exit' to quit.\n")

history = ""
while True:
    user_msg = input("You: ").strip()
    if user_msg.lower() in ("exit", "quit"):
        break

    # simple conversational framing -- the model learned this pattern from biodata.txt
    # if you wrote Q&A pairs like "Question: ... Answer: ..." in your data, mirror that here.
    prompt = f"{history}Question: {user_msg}\nAnswer:"
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    out = model.generate(idx, max_new_tokens=80, temperature=0.01, top_k=10)
    full_text = tok.decode(out[0].tolist())
    answer = full_text[len(prompt):].split("Question:")[0].strip()

    print(f"Bot: {answer}\n")
    history += f"Question: {user_msg}\nAnswer: {answer}\n"