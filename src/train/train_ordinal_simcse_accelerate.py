import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import os
import argparse
from sklearn.model_selection import train_test_split
from accelerate import Accelerator

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Ordinal Supervised SimCSE Training")
parser.add_argument("--json_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./models/ordinal_simcse")
parser.add_argument(
    "--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2"
)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accum_steps", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# -----------------------------
# Dataset
# -----------------------------
class SimCSEDataset(Dataset):
    def __init__(self, data):
        self.text1 = [d["sentence1"] for d in data]
        self.text2 = [d["sentence2"] for d in data]
        # Ordinal similarity: 1 - normalized label distance
        self.labels = [1 - abs(d["label1"] - d["label2"]) / 4 for d in data]

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        return {
            "text1": self.text1[idx],
            "text2": self.text2[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# -----------------------------
# Load & split dataset
# -----------------------------
with open(args.json_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

train_data, val_data = train_test_split(
    all_data, test_size=args.test_size, random_state=args.seed
)
print(f"Train pairs: {len(train_data)}, Validation pairs: {len(val_data)}")

train_dataset = SimCSEDataset(train_data)
val_dataset = SimCSEDataset(val_data)

# -----------------------------
# Accelerator setup
# -----------------------------
accelerator = Accelerator(mixed_precision="fp16")  # multi-GPU + mixed precision

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

optimizer = AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Prepare everything with accelerator
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=False)
    for step, batch in enumerate(loop):
        inputs1 = tokenizer(
            batch["text1"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_len,
        )
        inputs2 = tokenizer(
            batch["text2"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_len,
        )

        # Move tensors to device
        inputs1 = {k: v.to(accelerator.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(accelerator.device) for k, v in inputs2.items()}
        labels = batch["label"].to(accelerator.device)

        with accelerator.autocast():
            out1 = model(**inputs1).last_hidden_state[:, 0]
            out2 = model(**inputs2).last_hidden_state[:, 0]
            cos_sim = F.cosine_similarity(out1, out2)
            loss = F.mse_loss(cos_sim, labels)

        accelerator.backward(loss)

        if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
            optimizer.step()
            scheduler.step()
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
            inputs1 = tokenizer(
                batch["text1"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=args.max_len,
            )
            inputs2 = tokenizer(
                batch["text2"],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=args.max_len,
            )

            inputs1 = {k: v.to(accelerator.device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(accelerator.device) for k, v in inputs2.items()}
            labels = batch["label"].to(accelerator.device)

            out1 = model(**inputs1).last_hidden_state[:, 0]
            out2 = model(**inputs2).last_hidden_state[:, 0]
            cos_sim = F.cosine_similarity(out1, out2)
            loss = F.mse_loss(cos_sim, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(
        f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    if accelerator.is_main_process:
        epoch_dir = os.path.join(args.output_dir, f"epoch{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

print("Training complete!")
