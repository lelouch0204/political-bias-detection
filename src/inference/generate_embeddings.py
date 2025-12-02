import argparse
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import os


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(args.output_emb), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_labels), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    with open(args.test_pkl, "rb") as f:
        df_test = pickle.load(f)

    texts = df_test[args.text_field].tolist()
    labels = df_test[args.label_field].tolist()

    batch_size = args.batch_size
    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            out = model(**enc)
            e = out.last_hidden_state[:, 0]

        embs.append(e.cpu().numpy())

    X_test_emb = np.concatenate(embs, axis=0)
    y_test = np.array(labels)

    np.save(args.output_emb, X_test_emb)

    with open(args.output_labels, "wb") as f:
        pickle.dump(y_test, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_pkl", type=str, required=True)
    parser.add_argument("--text_field", type=str, default="cleaned_text")
    parser.add_argument("--label_field", type=str, default="label")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_emb", type=str, default="X_test_emb.npy")
    parser.add_argument("--output_labels", type=str, default="y_test.pkl")

    args = parser.parse_args()
    main(args)
