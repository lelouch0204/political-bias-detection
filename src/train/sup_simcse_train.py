#!/usr/bin/env python
# coding: utf-8

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import os
import argparse
import random
from collections import defaultdict

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Ordinal Supervised SimCSE Training")
parser.add_argument("--json_path", type=str, required=True, help="Path to JSON dataset")
parser.add_argument("--output_dir", type=str, default="./models/ordinal_simcse")
parser.add_argument("--model_name", type=str, default="roberta-base")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accum_steps", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--temp", type=float, default=0.05, help="Temperature for contrastive loss")
parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
class WeightedPairDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.groups = defaultdict(list)
        for d in data:
            self.groups[d['label1']].append(d)
        self.labels = sorted(self.groups.keys())

        # Precompute pairs
        self.pairs = []
        for label1 in self.labels:
            for d1 in self.groups[label1]:
                # positive pair
                self.pairs.append((d1, d1))
                # negative pairs weighted by distance
                for label2 in self.labels:
                    if label1 == label2:
                        continue
                    weight = abs(label1 - label2)  # bigger distance -> more important
                    for _ in range(weight):
                        d2 = random.choice(self.groups[label2])
                        self.pairs.append((d1, d2))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d1, d2 = self.pairs[idx]
        label_sim = 1 - abs(d1["label1"] - d2["label2"]) / 4
        return {
            "text1": d1["sentence1"],
            "text2": d2["sentence2"],
            "label": torch.tensor(label_sim, dtype=torch.float),
        }

# -----------------------------
# Load dataset
# -----------------------------
with open(args.json_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# simple train/val split
split_idx = int(len(all_data) * (1 - args.test_size))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

train_dataset = WeightedPairDataset(train_data)
val_dataset = WeightedPairDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# -----------------------------
# Model & Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name).to(args.device)
optimizer = AdamW(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler()

# -----------------------------
# Contrastive loss function
# -----------------------------
def contrastive_loss(out1, out2, label, temp=0.05):
    out1 = F.normalize(out1, dim=-1)
    out2 = F.normalize(out2, dim=-1)
    cos_sim = F.cosine_similarity(out1, out2)
    # scale to temperature
    cos_sim = cos_sim / temp
    # label is in [0,1] -> we scale as similarity target
    return F.mse_loss(cos_sim, label)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=False)
    optimizer.zero_grad()

    for step, batch in enumerate(loop):
        inputs1 = tokenizer(batch["text1"], padding=True, truncation=True,
                            return_tensors="pt", max_length=args.max_len).to(args.device)
        inputs2 = tokenizer(batch["text2"], padding=True, truncation=True,
                            return_tensors="pt", max_length=args.max_len).to(args.device)
        labels = batch["label"].to(args.device)

        with torch.cuda.amp.autocast():
            out1 = model(**inputs1).last_hidden_state[:, 0, :]
            out2 = model(**inputs2).last_hidden_state[:, 0, :]
            loss = contrastive_loss(out1, out2, labels, temp=args.temp)

        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs1 = tokenizer(batch["text1"], padding=True, truncation=True,
                                return_tensors="pt", max_length=args.max_len).to(args.device)
            inputs2 = tokenizer(batch["text2"], padding=True, truncation=True,
                                return_tensors="pt", max_length=args.max_len).to(args.device)
            labels = batch["label"].to(args.device)

            out1 = model(**inputs1).last_hidden_state[:, 0, :]
            out2 = model(**inputs2).last_hidden_state[:, 0, :]
            loss = contrastive_loss(out1, out2, labels, temp=args.temp)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
        epoch_dir = os.path.join(args.output_dir, f"epoch{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

print("Training complete!")