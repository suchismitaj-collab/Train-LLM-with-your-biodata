"""
Step 3: Train a small Byte-Pair-Encoding tokenizer on your own biodata.
This is the same technique GPT-2/GPT-3 use, just trained on your tiny corpus.
"""
from tokenizers import ByteLevelBPETokenizer
import os

DATA_PATH = "biodata.txt"
VOCAB_SIZE = 200 # small because your dataset is small; bump up if you add way more text
OUT_DIR = "tokenizer"

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[DATA_PATH],
    vocab_size=VOCAB_SIZE,
    min_frequency=1,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)
tokenizer.save_model(OUT_DIR)
print(f"Tokenizer saved to {OUT_DIR}/ (vocab.json, merges.txt)")