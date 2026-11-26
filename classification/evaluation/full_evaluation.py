import os
import torch

from evaluation.metrics import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    compute_basic_metrics,
    compute_multiclass_roc,
    plot_multiclass_roc,
    compute_multiclass_pr,
    plot_multiclass_pr,
)

from evaluation.embedding import collect_latents, plot_umap
from evaluation.cam import GradCAM1D, generate_cam_for_sample, plot_cam


def run_full_evaluation(model, datamodule, run_dir, cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    test_loader = datamodule.test_dataloader()

    # --- Confusion Matrix ---
    cm = compute_confusion_matrix(model, test_loader, device=device)
    plot_confusion_matrix(
        cm,
        normalize=False,
        save_path=os.path.join(fig_dir, "confusion_matrix.png")
    )

    # --- Basic Metrics ---
    compute_basic_metrics(
        model,
        test_loader,
        device=device,
        save_path=os.path.join(fig_dir, "test_metrics.json")
    )

    # --- ROC ---
    fpr, tpr, roc_auc = compute_multiclass_roc(model, test_loader)
    plot_multiclass_roc(
        fpr,
        tpr,
        roc_auc,
        class_names=["Random", "Core-Shell", "Ordered"],
        save_path=os.path.join(fig_dir, "roc_curve.png")
    )

    # --- PR ---
    precision, recall, ap = compute_multiclass_pr(model, test_loader)
    plot_multiclass_pr(
        precision,
        recall,
        ap,
        class_names=["Random", "Core-Shell", "Ordered"],
        save_path=os.path.join(fig_dir, "pr_curve.png")
    )

    # --- UMAP ---
    latents, labels = collect_latents(model, test_loader)
    plot_umap(
        latents.numpy(),
        labels.numpy(),
        save_path=os.path.join(fig_dir, "umap.png")
    )

    # --- GradCAM ---
    batch = next(iter(test_loader))

    if cfg.model_type == "onebranch":
        target_layer = model.model.branch.resblocks[1].block[3]
    else:
        target_layer = model.model.xanes_branch.resblocks[1].block[3]

    cam = GradCAM1D(model, target_layer)
    cams = cam.generate_from_batch(batch)

    xanes = batch[0][0].cpu().numpy()
    plot_cam(
        xanes,
        cams[0].cpu().numpy(),
        save_path=os.path.join(fig_dir, "CAM.png")
    )

    cam.remove_hooks()
