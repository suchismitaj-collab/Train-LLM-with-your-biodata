"""
Step 5: Train MiniGPT from scratch on your biodata.txt using the tokenizer
trained in step 3. Saves checkpoint to checkpoints/mini_llm.pt
"""
import os
import torch
from tokenizers import ByteLevelBPETokenizer
from model import MiniGPT

# ---- config ----
DATA_PATH = "biodata.txt"
TOKENIZER_DIR = "tokenizer"
CKPT_DIR = "checkpoints"
BLOCK_SIZE = 100
BATCH_SIZE = 8
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
LR = 3e-4
MAX_ITERS = 4000
EVAL_INTERVAL = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CKPT_DIR, exist_ok=True)

# ---- load tokenizer ----
tok = ByteLevelBPETokenizer(
    f"{TOKENIZER_DIR}/vocab.json", f"{TOKENIZER_DIR}/merges.txt"
)
vocab_size = tok.get_vocab_size()
print(f"Vocab size: {vocab_size}")

# ---- load + encode data ----
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

ids = tok.encode(text).ids
data = torch.tensor(ids, dtype=torch.long)
n = int(0.85 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Total tokens: {len(data)} (train {len(train_data)}, val {len(val_data)})")

if len(train_data) < BLOCK_SIZE + 1:
    raise ValueError(
        "Your biodata.txt is too short for this block_size. "
        "Add more text or reduce BLOCK_SIZE in this script."
    )
if len(val_data) < BLOCK_SIZE + 1:
    raise ValueError(
        "Your validation split is too short for this block_size. "
        "Add more text to biodata.txt, or reduce BLOCK_SIZE."
    )


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


model = MiniGPT(vocab_size, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"Training on {DEVICE} | params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

for it in range(MAX_ITERS):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch("val")
            _, vloss = model(xv, yv)
        model.train()
        print(f"iter {it:5d} | train loss {loss.item():.4f} | val loss {vloss.item():.4f}")

torch.save(
    {
        "model_state": model.state_dict(),
        "config": dict(
            vocab_size=vocab_size, block_size=BLOCK_SIZE,
            n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
        ),
    },
    f"{CKPT_DIR}/mini_llm.pt",
)
print(f"Saved checkpoint to {CKPT_DIR}/mini_llm.pt")