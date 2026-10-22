"""
split_dataset.py
================
Splits dataset.json into train, validation, and test sets (80/10/10 ratio)
with a fixed random seed for exact reproducibility.

Outputs:
  - train.json (~80% of data)
  - val.json   (~10% of data)
  - test.json  (~10% of data)
"""

import json
import random

DATASET_FILE = "dataset.json"
TRAIN_FILE = "train.json"
VAL_FILE = "val.json"
TEST_FILE = "test.json"
SEED = 42

def split_data():
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total samples in {DATASET_FILE}: {len(data)}")

    random.seed(SEED)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"Split complete:")
    print(f"  - Train: {len(train_data)} samples -> {TRAIN_FILE}")
    print(f"  - Validation: {len(val_data)} samples -> {VAL_FILE}")
    print(f"  - Test: {len(test_data)} samples -> {TEST_FILE}")

if __name__ == "__main__":
    split_data()
