import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_classification_report(y_true, y_pred, class_names, out_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']

    data = []
    for class_name in class_names:
        for metric in metrics:
            data.append({'Class': class_name, 'Metric': metric.capitalize(), 'Value': report[class_name][metric]})

    df_plot = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Class', y='Value', hue='Metric')
    plt.ylim(0, 1)
    plt.title('Classification Metrics per Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, class_names, out_path):
    y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_binarized.shape[1]

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def load_model_and_tokenizer(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict(model, tokenizer, texts, max_length=256):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy(), logits.cpu().numpy()


def preprocess_from_hf(dataset_name, min_text_length=50, test_size=0.2, dataset_config=None):
    ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
    split = 'train' if 'train' in ds else list(ds.keys())[0]
    df = ds[split].to_pandas().reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated()]

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
    df['text_input'] = df['clean_text'].fillna(df['Text'])
    df = df[df['text_input'].str.len() > min_text_length]

    X_text = df['text_input'].values
    y = df['bias_label'].values
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train_text, X_test_text, y_train, y_test


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model and save figures")
    parser.add_argument('--model_path', type=str, required=True, help='Path to HF model directory')
    parser.add_argument('--dataset_name', type=str, default='lelouch0204/cleaned_allsides_v2.csv', help='HF dataset to load')
    parser.add_argument('--dataset_config', type=str, default=None, help='HF dataset config/subset name')
    parser.add_argument('--output_dir', type=str, default='./outputs/eval_methods', help='Directory to save figures and reports')
    parser.add_argument('--min_text_length', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--max_length', type=int, default=256)
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading and preprocessing dataset: {args.dataset_name}")
    X_train_text, X_test_text, y_train, y_test = preprocess_from_hf(
        dataset_name=args.dataset_name,
        min_text_length=args.min_text_length,
        test_size=args.test_size,
        dataset_config=args.dataset_config,
    )
    print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(os.path.expanduser(args.model_path))

    print("Predicting on test set...")
    preds, logits = predict(model, tokenizer, X_test_text.tolist(), max_length=args.max_length)

    # Softmax probabilities for ROC curves
    proba = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    report = classification_report(y_test, preds, digits=4)
    print(report)

    # Save metrics and report
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    # Save confusion matrix figure
    class_names = ['Right', 'Lean Right', 'Center', 'Lean Left', 'Left']
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, preds, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Save classification report bar chart
    cr_path = os.path.join(out_dir, 'classification_report_bars.png')
    plot_classification_report(y_test, preds, class_names, cr_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Save ROC curve
    roc_path = os.path.join(out_dir, 'roc_curve.png')
    plot_roc_curve(y_test, proba, class_names, roc_path)
    print(f"Saved ROC curve to {roc_path}")

    print(f"Outputs saved in {out_dir}")


if __name__ == '__main__':
    main()
