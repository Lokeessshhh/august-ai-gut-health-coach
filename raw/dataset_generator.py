import json
import random
import os

# Input and output paths
INPUT_FILE = "data.jsonl"       # your JSONL dataset
OUTPUT_DIR = "dataset"          # output directory

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSONL dataset
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # skip empty lines
            data.append(json.loads(line))

# Shuffle dataset
random.shuffle(data)

# Split ratios
train_split = 0.8
val_split = 0.1
test_split = 0.1

n_total = len(data)
n_train = int(n_total * train_split)
n_val = int(n_total * val_split)

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Total: {n_total}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# Helper to save JSONL
def save_jsonl(data_list, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Save files
save_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
save_jsonl(val_data, os.path.join(OUTPUT_DIR, "val.jsonl"))
save_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))

print(f"Files saved in {OUTPUT_DIR}/")
