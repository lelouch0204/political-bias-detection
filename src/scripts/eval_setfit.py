import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from setfit import SetFitModel


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
            data.append({
                'Class': class_name,
                'Metric': metric.capitalize(),
                'Value': report[class_name][metric]
            })

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


def load_and_preprocess(dataset_name=None, data_file=None, min_text_length=50, test_size=0.2):
    if dataset_name:
        ds = load_dataset(dataset_name)
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
    elif data_file:
        if data_file.endswith('.pkl') or data_file.endswith('.pickle'):
            df = pd.read_pickle(data_file)
        elif data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
    else:
        raise ValueError("Provide either dataset_name (HF) or data_file (local)")

    df = df.reset_index(drop=True)
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
    parser = argparse.ArgumentParser(description="Evaluate a saved SetFit model")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory of saved SetFit model')
    parser.add_argument('--dataset_name', type=str, default='lelouch0204/cleaned_allsides_v2.csv', help='HF dataset to load')
    parser.add_argument('--data_file', type=str, default=None, help='Local data file (.pkl or .csv) as alternative to HF')
    parser.add_argument('--output_dir', type=str, default='./outputs/eval_setfit', help='Directory to save figures and reports')
    parser.add_argument('--min_text_length', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("Loading and preprocessing data...")
    X_train_text, X_test_text, y_train, y_test = load_and_preprocess(
        dataset_name=args.dataset_name if args.data_file is None else None,
        data_file=os.path.expanduser(args.data_file) if args.data_file else None,
        min_text_length=args.min_text_length,
        test_size=args.test_size,
    )
    print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

    # Load model
    model_dir = os.path.expanduser(args.model_dir)
    print(f"Loading SetFit model from {model_dir}")
    model = SetFitModel.from_pretrained(model_dir)

    # Predict
    print("Running predictions on test set...")
    preds = model.predict(list(X_test_text))
    # SetFit exposes predict_proba for probabilities (for ROC curves)
    proba = model.predict_proba(list(X_test_text))

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    report = classification_report(y_test, preds, digits=4)
    print(report)

    # Save report
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\nF1-macro: {f1:.4f}\n\n")
        f.write(report)

    # Confusion matrix figure
    class_names = ['right', 'lean right', 'center', 'lean left', 'left']
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, preds, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    # Classification report bar chart
    cr_path = os.path.join(out_dir, 'classification_report_bars.png')
    plot_classification_report(y_test, preds, class_names, cr_path)
    print(f"Saved classification report bars to {cr_path}")

    # ROC curve
    roc_path = os.path.join(out_dir, 'roc_curve.png')
    plot_roc_curve(y_test, proba, class_names, roc_path)
    print(f"Saved ROC curve to {roc_path}")

    print(f"Outputs saved in {out_dir}")


if __name__ == '__main__':
    main()
