import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.metrics import format_class_labels

# Okabe-Ito colorblind-safe palette
_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']

_RC = {
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '0.8',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
}


def plot_publish_figure(
    cm,
    latent_vectors,
    latent_labels,
    precision,
    recall,
    ap,
    fpr,
    tpr,
    roc_auc,
    class_names=None,
    normalize_cm=False,
    save_path="publish_fig.png",
):
    """
    4×1 publication-quality composite figure saved at dpi=600.

    Panels
    ------
    (a) Confusion matrix
    (b) UMAP of latent space
    (c) Precision–Recall curves (OvR)
    (d) ROC curves (OvR)

    Parameters
    ----------
    cm             : ndarray (C, C)
    latent_vectors : ndarray (N, D)
    latent_labels  : ndarray (N,)    integer class indices
    precision      : dict {i: array}
    recall         : dict {i: array}
    ap             : dict {i: float}
    fpr            : dict {i: array}
    tpr            : dict {i: array}
    roc_auc        : dict {i: float}
    class_names    : list[str], optional
    normalize_cm   : bool  – row-normalise confusion matrix
    save_path      : str
    """
    import umap as umap_lib

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    class_names = format_class_labels(class_names)
    colors = _COLORS[:n_classes]

    # Normalize binary (scalar/array) inputs to dict format expected by the loops below
    if not isinstance(precision, dict):
        precision = {1: precision}
        recall = {1: recall}
        ap = {1: ap}
    if not isinstance(fpr, dict):
        fpr = {1: fpr}
        tpr = {1: tpr}
        roc_auc = {1: roc_auc}
    curve_indices = sorted(precision.keys())

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(4, 1, figsize=(5.5, 18))
        fig.subplots_adjust(hspace=0.55)

        # ── (a) Confusion Matrix ──────────────────────────────────────────
        ax = axes[0]
        cm_plot = (
            cm.astype(float) / cm.sum(axis=1, keepdims=True)
            if normalize_cm
            else cm.astype(float)
        )

        im = ax.imshow(
            cm_plot, interpolation='nearest', cmap='Blues',
            vmin=0, vmax=(1.0 if normalize_cm else cm_plot.max()),
        )
        ax.set_aspect('auto', adjustable='box', anchor='W')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
        cbar.ax.tick_params(labelsize=7)
        if normalize_cm:
            cbar.set_label('Fraction', fontsize=8)

        ticks = np.arange(n_classes)
        ax.set_xticks(ticks)
        ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
        ax.set_yticks(ticks)
        ax.set_yticklabels(class_names, fontsize=9)

        thresh = cm_plot.max() / 2.0
        fmt = '.2f' if normalize_cm else 'g'
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                ax.text(
                    j, i, format(cm_plot[i, j], fmt),
                    ha='center', va='center', fontsize=9,
                    color='white' if cm_plot[i, j] > thresh else 'black',
                )
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title('Confusion Matrix')

        # ── (b) UMAP ─────────────────────────────────────────────────────
        ax = axes[1]
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(latent_vectors)

        for idx, c in enumerate(np.unique(latent_labels)):
            mask = latent_labels == c
            name = class_names[int(c)] if int(c) < len(class_names) else f"Class {c}"
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=name, s=14, alpha=0.75,
                color=colors[idx % len(colors)], edgecolors='none', linewidths=0,
            )
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Latent Space (UMAP)')
        ax.legend(loc='best', markerscale=1.8, handletextpad=0.3, borderpad=0.5)
        ax.grid(True, alpha=0.2, linewidth=0.4)

        # ── (c) Precision–Recall Curve ────────────────────────────────────
        ax = axes[2]
        for i in curve_indices:
            ax.plot(
                recall[i], precision[i],
                color=colors[i % len(colors)],
                label=f"{class_names[i]}  AP = {ap[i]:.3f}",
            )
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision–Recall Curve (OvR)')
        ax.legend(loc='lower left', borderpad=0.5)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # ── (d) ROC Curve ─────────────────────────────────────────────────
        ax = axes[3]
        for i in curve_indices:
            ax.plot(
                fpr[i], tpr[i],
                color=colors[i % len(colors)],
                label=f"{class_names[i]}  AUC = {roc_auc[i]:.3f}",
            )
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.9, color='#888888', label='Chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (OvR)')
        ax.legend(loc='lower right', borderpad=0.5)
        ax.grid(True, alpha=0.2, linewidth=0.4)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        # ── Panel labels (a) – (d) ────────────────────────────────────────
        for ax, label in zip(axes, ['(a)', '(b)', '(c)', '(d)']):
            ax.text(
                -0.13, 1.06, label,
                transform=ax.transAxes,
                fontsize=12, fontweight='bold',
                va='bottom', ha='left',
                clip_on=False,
            )

        if save_path is not None:
            fig.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved → {save_path}")

        plt.close(fig)
