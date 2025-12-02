import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import wandb

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert weights to tensor and move to device
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Feed inputs to model
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move weights to the same device as the model
        if self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)
            
        # Standard Cross Entropy but with weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SimCSE model on AllSides data")
    
    # REQUIRED: Path to your SimCSE model
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the pre-trained SimCSE model"
    )
    
    # Dataset from Hugging Face
    parser.add_argument("--dataset_name", type=str, default="tasksource/allsides", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config/subset name")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")

    # Optional hyperparameters
    parser.add_argument("--output_dir", type=str, default="./allsides_finetuned")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--min_text_length", type=int, default=50, help="Minimum text length filter")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "cs7641-allsides"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"), choices=["online", "offline", "disabled"])

    args = parser.parse_args()
    
    # Expand paths (fix '~' error)
    args.model_path = os.path.expanduser(args.model_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    
    return args

def load_file_to_dataset(file_path):
    """Helper to load .pkl or .csv into a HuggingFace Dataset"""
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        df = pd.read_pickle(file_path)
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicated columns if any
        return Dataset.from_pandas(df)
    elif file_path.endswith('.csv'):
        return Dataset.from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

def main():
    args = parse_args()
    
    # ================= 1. INIT W&B =================
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args)
    )
    
    # ================= 2. LOAD & PREPROCESS DATA =================
    print(f"Loading dataset from HuggingFace: {args.dataset_name}")
    
    # Load data from HuggingFace
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
    else:
        dataset = load_dataset(args.dataset_name)
    
    # Assume dataset has 'train' split, or use the first available split
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        # Use first available split
        split_name = list(dataset.keys())[0]
        print(f"No 'train' split found, using '{split_name}'")
        df = dataset[split_name].to_pandas()
    
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Apply new bias mapping (reversed scale)
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
    df = df[df['text_input'].str.len() > args.min_text_length]
    
    print(f"Final dataset: {len(df)} samples")
    print("Bias distribution:", df['bias_label'].value_counts().sort_index().to_dict())
    wandb.log({"dataset_size": len(df), "bias_distribution": df['bias_label'].value_counts().to_dict()})
    
    # Train-test split
    X_text = df['text_input'].values
    y = df['bias_label'].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")
    wandb.log({"train_size": len(X_train_text), "test_size": len(X_test_text)})
    
    # Create datasets
    train_dataset = Dataset.from_dict({'text_input': X_train_text, 'label': y_train})
    val_dataset = Dataset.from_dict({'text_input': X_test_text, 'label': y_test})
    
    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print(f"Computed Class Weights: {class_weights}")
    wandb.config.update({"class_weights": class_weights.tolist()})

    # ================= 3. TOKENIZATION =================
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    def tokenize_function(examples):
        return tokenizer(
            examples['text_input'], 
            truncation=True, 
            padding="max_length", 
            max_length=args.max_length
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # ================= 4. MODEL SETUP =================
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    
    # Watch model with W&B
    wandb.watch(model, log="all", log_freq=100)

    # ================= 5. TRAINER SETUP =================
    for param in model.base_model.parameters():
        param.requires_grad = False  # Train only the classification head

    training_args_1 = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "phase1"),
        learning_rate=1e-3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=os.path.join(args.output_dir, 'logs'),
        save_total_limit=1,
        report_to="wandb" if args.wandb_mode != "disabled" else "none"
    )

    trainer_phase1 = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args_1,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer_phase1.train()
    print("\nEvaluation after Phase 1:")
    metrics = trainer_phase1.evaluate()
    print(metrics)
    wandb.log({"phase1/accuracy": metrics.get("eval_accuracy"), "phase1/f1_macro": metrics.get("eval_f1_macro")})
    trainer_phase1.save_model(os.path.join(args.output_dir, "phase1_model"))
    
    for param in model.base_model.parameters():
        param.requires_grad = True  # Unfreeze all layers for second phase
    
    training_args_2 = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "stage2"),
        learning_rate=args.lr,       # Low LR to preserve features
        num_train_epochs=args.epochs,       # Just a few epochs of polish
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        report_to="wandb" if args.wandb_mode != "disabled" else "none"
    )
    
    trainer_phase2 = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args_2,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting Fine-Tuning...")
    trainer_phase2.train()

    print("\nFinal Evaluation:")
    metrics = trainer_phase2.evaluate()
    print(metrics)
    wandb.log({"final/accuracy": metrics.get("eval_accuracy"), "final/f1_macro": metrics.get("eval_f1_macro")})
    
    trainer_phase2.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Log model as W&B artifact
    try:
        artifact = wandb.Artifact(
            name=f"finetuned-allsides-{wandb.run.id}",
            type="model",
            metadata={
                "final_accuracy": metrics.get("eval_accuracy"),
                "final_f1_macro": metrics.get("eval_f1_macro"),
                "num_epochs_phase1": 10,
                "num_epochs_phase2": args.epochs
            }
        )
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact)
        print(f"Model artifact logged to W&B")
    except Exception as e:
        print(f"Failed to log artifact: {e}")
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()