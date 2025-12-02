import os
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import wandb

parser = argparse.ArgumentParser(description="Train SetFit with W&B logging")
parser.add_argument("--model_name", default="sentence-transformers/all-mpnet-base-v2", type=str)
parser.add_argument("--train_pickle", default="./data/allsides/dataset_train_split.pkl", type=str)
parser.add_argument("--test_pickle", default="./data/allsides/dataset_test_split.pkl", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_epochs", default=1, type=int)
parser.add_argument("--num_iterations", default=20, type=int)
parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "cs7641-allsides"), type=str)
parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", None), type=str)
parser.add_argument("--wandb_run_name", default=None, type=str)
parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"), choices=["online","offline","disabled"], type=str)
args = parser.parse_args()

# 1. Load Data
# train_df = pd.read_pickle(args.train_pickle).reset_index(drop=True)
# test_df = pd.read_pickle(args.test_pickle).reset_index(drop=True)
print("Loading dataset...")
ds = load_dataset("lelouch0204/cleaned_allsides_v2.csv")
df = ds['train'].to_pandas()
print(f"Loaded: {len(df)} samples")

bias_mapping = {
    'left': 4,
    'lean left': 3,
    'center': 2,
    'lean right': 1,
    'right': 0
}
df['bias_label'] = df['Bias'].str.lower().str.strip().map(bias_mapping)
df = df.dropna(subset=['bias_label', 'Text'])
df['bias_label'] = df['bias_label'].astype(int)

# Use existing clean_text or Text
df['text_input'] = df['clean_text'].fillna(df['Text'])
df = df[df['text_input'].str.len() > 50]

print(f"Final dataset: {len(df)} samples")
print("Bias distribution:", df['bias_label'].value_counts().sort_index().to_dict())

# Train-test split
X_text = df['text_input'].values
y = df['bias_label'].values
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")



# # Clean labels (Map text to int if needed)
# label_map = {'left': 0, 'lean left': 1, 'center': 2, 'lean right': 3, 'right': 4}
# def clean_label(x):
#     if isinstance(x, str): return label_map.get(x.lower().strip(), -1)
#     return int(x)

# train_df['label'] = train_df['Bias'].apply(clean_label)
# test_df['label'] = test_df['Bias'].apply(clean_label)

# Convert to HuggingFace Dataset
train_ds = Dataset.from_pandas(pd.DataFrame({"text": X_train_text, "label": y_train}))
test_ds = Dataset.from_pandas(pd.DataFrame({"text": X_test_text, "label": y_test}))
# 2. Init W&B
wandb.init(
    project=args.wandb_project,
    entity=args.wandb_entity,
    name=args.wandb_run_name,
    mode=args.wandb_mode,
    config={
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "num_iterations": args.num_iterations,
        "train_size": len(train_ds) if 'train_ds' in locals() else None,
        "test_size": len(test_ds) if 'test_ds' in locals() else None,
    },
)

# 3. Load efficient Sentence Transformer
# 'all-mpnet-base-v2' is much stronger than roberta-base for embeddings
MODEL_NAME = args.model_name
model = SetFitModel.from_pretrained(MODEL_NAME)

# 3. Train
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    metric="accuracy",
    batch_size=args.batch_size,
    num_epochs=args.num_epochs, # SetFit converges extremely fast (often 1 epoch is enough)
    num_iterations=args.num_iterations, # Number of pairs to generate per sentence
    # column_mapping={"Text": "text", "label": "label"},
)

print("Training SetFit Model...")
trainer.train()

# Best effort: watch model weights (SetFit wraps a sentence-transformers backbone)
try:
    wandb.watch(model.model_body if hasattr(model, "model_body") else model, log="all")
except Exception:
    pass

# 4. Evaluate
metrics = trainer.evaluate()
print(f"Final Metrics: {metrics}")
wandb.log({"eval/accuracy": metrics.get("accuracy", None)})

# 5. Save
OUTPUT_DIR = os.path.expanduser(f"~/scratch/experiments/setfit_updated_train_test/{args.model_name}/_{args.num_epochs}_epochs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)

# Log saved model as W&B artifact
try:
    artifact = wandb.Artifact(
        name=f"setfit-{args.model_name.replace('/','-')}",
        type="model",
        metadata={"num_epochs": args.num_epochs, "batch_size": args.batch_size}
    )
    artifact.add_dir(OUTPUT_DIR)
    wandb.log_artifact(artifact)
except Exception:
    pass

wandb.finish()