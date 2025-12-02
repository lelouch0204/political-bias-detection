# unsup_simcse_train.py
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_file", type=str, required=True, help="one-sentence-per-line file"
    )
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--output_dir", type=str, default="./simcse_unsup")
    p.add_argument(
        "--batch_size", type=int, default=128
    )  # effective batch; lower if OOM
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument(
        "--use_cls", action="store_true", help="use [CLS] pooling instead of mean"
    )
    p.add_argument(
        "--proj_dim",
        type=int,
        default=0,
        help="projection head dim; 0 disables (use encoder output directly)",
    )
    return p.parse_args()


def build_dataloader(train_file, tokenizer, max_length, batch_size):
    ds = load_dataset("text", data_files={"train": train_file})["train"]

    def tokenize(ex):
        t = tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        return t

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def mean_pooling(token_embeds, attn_mask):
    attn_mask = attn_mask.unsqueeze(-1)
    summed = (token_embeds * attn_mask).sum(1)
    counts = attn_mask.sum(1).clamp(min=1e-9)
    return summed / counts


def cls_pooling(token_embeds):
    return token_embeds[:, 0]


def simcse_loss(embeddings, tau):
    # embeddings: (2N, d), where embeddings are [e1_1, e1_2, ..., e1_N, e2_1, e2_2, ..., e2_N]
    device = embeddings.device
    N2 = embeddings.size(0)
    assert N2 % 2 == 0
    N = N2 // 2
    # normalize
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    # similarity matrix
    sim = torch.matmul(emb_norm, emb_norm.T) / tau  # (2N,2N)
    # mask out self-similarity
    diag = torch.eye(N2, device=device).bool()
    sim.masked_fill_(diag, -1e12)

    # targets: for i in [0..2N-1], positive index is i^1 (flip last bit)
    target = torch.arange(N2, device=device) ^ 1
    loss = F.cross_entropy(sim, target)
    return loss


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.train()

    # Optional projection head improves contrastive learning stability
    proj = None
    if args.proj_dim and args.proj_dim > 0:
        hidden = model.config.hidden_size
        proj = torch.nn.Sequential(
            torch.nn.Linear(hidden, args.proj_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(args.proj_dim, args.proj_dim),
        ).to(device)

    dataloader = build_dataloader(
        args.train_file, tokenizer, args.max_length, args.batch_size
    )
    params = list(model.parameters()) + (
        list(proj.parameters()) if proj is not None else []
    )
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs // args.accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            # batch contains input_ids and attention_mask with shape (B, L)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            # Forward pass 1
            out1 = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            last1 = out1.last_hidden_state  # (B, L, d)
            emb1 = (
                cls_pooling(last1) if args.use_cls else mean_pooling(last1, attn_mask)
            )  # (B, d)
            if proj is not None:
                emb1 = proj(emb1)

            # Forward pass 2 (dropout different)
            out2 = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            last2 = out2.last_hidden_state
            emb2 = (
                cls_pooling(last2) if args.use_cls else mean_pooling(last2, attn_mask)
            )
            if proj is not None:
                emb2 = proj(emb2)

            embeddings = torch.stack((emb1, emb2), dim=1)  # (B, 2, d)
            embeddings = embeddings.view(-1, embeddings.size(-1))  # (2B, d)
            loss = simcse_loss(embeddings, args.tau)
            loss = loss / args.accum_steps
            loss.backward()

            if (step + 1) % args.accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": loss.item()})

    # Save the encoder
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)


if __name__ == "__main__":
    main()
