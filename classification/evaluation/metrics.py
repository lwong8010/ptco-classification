import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# -------------------------------------------------------------
# Class name display formatting
# -------------------------------------------------------------

def format_class_label(name):
    """Render ratio-style class names like '1to1' as '1:1' for display.

    Names that don't match the "<int>to<int>" pattern (e.g. "Random",
    "Core-Shell") are returned unchanged.
    """
    match = re.fullmatch(r"(\d+)to(\d+)", str(name))
    return f"{match.group(1)}:{match.group(2)}" if match else name

def format_class_labels(class_names):
    """Apply format_class_label to a list of class names, if provided."""
    return [format_class_label(n) for n in class_names] if class_names else class_names

@torch.no_grad()
def collect_predictions(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []
    all_logits = []

    for batch_idx, batch in enumerate(dataloader):
        batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]

        out = model.predict_step(batch, batch_idx)

        all_preds.append(out["preds"].cpu())
        all_labels.append(out["labels"].cpu())
        all_probs.append(out["probs"].cpu())
        all_logits.append(out["logits"].cpu())

    return {
        "preds": torch.cat(all_preds),
        "labels": torch.cat(all_labels),
        "probs": torch.cat(all_probs),
        "logits": torch.cat(all_logits),
    }

# -------------------------------------------------------------
# Metric 1: Confusion Matrix
# -------------------------------------------------------------

from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(model, dataloader, device="cuda"):
    data = collect_predictions(model, dataloader, device)
    return confusion_matrix(data["labels"], data["preds"])

def plot_confusion_matrix(cm, class_names=None, normalize=False, save_path=None):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    class_names = format_class_labels(class_names)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    # Print numbers
    thresh = cm.max() / 5.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i,j]:.2f}" if normalize else f"{cm[i,j]}"
            plt.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm[i,j] < thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    
    plt.close()
    
# -------------------------------------------------------------
# Metric 2: Accuracy, Precision, F1
# -------------------------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def compute_basic_metrics(model, dataloader, device="cuda", save_path=None):
    data = collect_predictions(model, dataloader, device)

    y_true = data["labels"].numpy()
    y_pred = data["preds"].numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

# -------------------------------------------------------------
# Metric 3: ROC Curve + AUC
# -------------------------------------------------------------

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_multiclass_roc(model, dataloader, device="cuda:0"):
    model.to(device)
    model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = model.unpack_batch(batch)

            if "xanes" in inputs:
                inputs["xanes"] = inputs["xanes"].unsqueeze(1).to(device)
            if "exafs" in inputs:
                inputs["exafs"] = inputs["exafs"].unsqueeze(1).to(device)

            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).numpy()

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc

def plot_multiclass_roc(fpr, tpr, roc_auc, class_names=None, save_path=None):
    """
    fpr, tpr, roc_auc: dicts keyed by class index
    class_names: optional list like ["Class 0", "Class 1", ...]
    """
    plt.figure(figsize=(6, 6))

    n_classes = len(fpr)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    class_names = format_class_labels(class_names)

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})"
        )

    # Diagonal baseline
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC (OvR)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    
    plt.close()


# -------------------------------------------------------------
# Metric 4: Precision-Recall Curve (PR Curve) + Average Precision (AP)
# -------------------------------------------------------------

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

def compute_multiclass_pr(model, dataloader, device="cuda"):
    data = collect_predictions(model, dataloader, device)

    y_true = data["labels"].numpy()          # (N,)
    y_probs = data["probs"].numpy()          # (N, C)

    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    precision = {}
    recall = {}
    ap = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i],
            y_probs[:, i]
        )
        ap[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

    return precision, recall, ap

def plot_multiclass_pr(precision, recall, ap, class_names=None, save_path=None):
    plt.figure(figsize=(6, 6))

    n_classes = len(precision)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    class_names = format_class_labels(class_names)

    for i in range(n_classes):
        plt.plot(
            recall[i],
            precision[i],
            label=f"{class_names[i]} (AP = {ap[i]:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision–Recall Curve (OvR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    
    plt.close()

# -------------------------------------------------------------
# Metric 4b: Binary ROC and PR Curves
# -------------------------------------------------------------

def compute_binary_roc(model, dataloader, device="cuda"):
    from sklearn.metrics import roc_curve, auc

    data = collect_predictions(model, dataloader, device)
    y_true = data["labels"].numpy()
    y_probs = data["probs"].numpy()  # (N, 2)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(2):
        y_true_i = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_i, y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc


def plot_binary_roc(fpr, tpr, roc_auc, class_names=None, save_path=None):
    n_classes = len(fpr)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    class_names = format_class_labels(class_names)

    plt.figure(figsize=(6, 6))
    for i in sorted(fpr.keys()):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def compute_binary_pr(model, dataloader, device="cuda"):
    from sklearn.metrics import precision_recall_curve, average_precision_score

    data = collect_predictions(model, dataloader, device)
    y_true = data["labels"].numpy()
    y_probs = data["probs"].numpy()  # (N, 2)

    precision, recall, ap = {}, {}, {}
    for i in range(2):
        y_true_i = (y_true == i).astype(int)
        precision[i], recall[i], _ = precision_recall_curve(y_true_i, y_probs[:, i])
        ap[i] = average_precision_score(y_true_i, y_probs[:, i])
    return precision, recall, ap


def plot_binary_pr(precision, recall, ap, class_names=None, save_path=None):
    n_classes = len(precision)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    class_names = format_class_labels(class_names)

    plt.figure(figsize=(6, 6))
    for i in sorted(precision.keys()):
        plt.plot(recall[i], precision[i], label=f"{class_names[i]} (AP = {ap[i]:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# -------------------------------------------------------------
# Metric 5: Calibration Curve * NOT BUILT FOR MULTICLASS
# -------------------------------------------------------------

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(y_true, y_scores, n_bins=10, save_path=None):
    """
    Plots a calibration curve (reliability diagram).

    Parameters:
    - y_true: array-like, shape (n_samples,), ground truth labels {0,1}
    - y_scores: array-like, shape (n_samples,), predicted probabilities for class 1
    - n_bins: number of bins for the calibration curve
    """
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    plt.title("Calibration Curve (Reliability Diagram)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    
    plt.close()

    return prob_true, prob_pred
