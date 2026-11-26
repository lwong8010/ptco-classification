import os
import numpy as np
from glob import glob
from pathlib import Path

run_dir = Path("outputs/2026-11-04/10-28-59")

class_name_set = ["Random", "Core-Shell", "Ordered"]

ckpt_dir = os.path.join(run_dir, "checkpoints")
ckpt_files = glob(os.path.join(ckpt_dir, "*.ckpt"))
config_file = os.path.join(run_dir,".hydra/config.yaml")

# pick the first (or sort by name)
ckpt_path = sorted(ckpt_files)[-1]  # latest by filename
print("Using checkpoint:", ckpt_path)

from models.lightning_classifier import SpectraClassifierModule

# load model
model = SpectraClassifierModule.load_from_checkpoint(ckpt_path, weights_only=False)
model.eval()  # set to evaluation mode

fig_dir = os.path.join(run_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

print("Saving figures to:", fig_dir)

from omegaconf import OmegaConf

cfg = OmegaConf.load(config_file)

from data.spectra_datamodule import SpectraDataModule

datamodule = SpectraDataModule(cfg.data)
datamodule.load_splits_from_run(run_dir)

test_loader = datamodule.test_dataloader()

from evaluation.metrics import (
    compute_confusion_matrix,
    plot_confusion_matrix,
    compute_basic_metrics,
    compute_multiclass_roc,
    plot_multiclass_roc,
    compute_multiclass_pr,
    plot_multiclass_pr,
    compute_binary_roc,
    plot_binary_roc,
    compute_binary_pr,
    plot_binary_pr,
)

# Confusion matrix
cm = compute_confusion_matrix(model, test_loader, device="cuda")
plot_confusion_matrix(
    cm,
    class_names=class_name_set,
    normalize=False,
    save_path=os.path.join(fig_dir, "confusion_matrix.png")
)
np.save(os.path.join(fig_dir, "confusion_matrix.npy"), cm)

# Basic metrics
metrics = compute_basic_metrics(
    model,
    test_loader,
    device="cuda",
    save_path=os.path.join(fig_dir, "test_metrics.json")
)
print(metrics)

# ROC Curve + AUC
if cfg.model.num_classes == 2:
    fpr, tpr, roc_auc = compute_binary_roc(model, test_loader)
    plot_binary_roc(
        fpr,
        tpr,
        roc_auc,
        class_names=class_name_set,
        save_path=os.path.join(fig_dir, "roc_curve.png")
    )
else:
    fpr, tpr, roc_auc = compute_multiclass_roc(model, test_loader)
    plot_multiclass_roc(
        fpr,
        tpr,
        roc_auc,
        class_names=class_name_set,
        save_path=os.path.join(fig_dir, "roc_curve.png")
    )

# Precision-Recall Curve (PR Curve) + Average Precision (AP)
if cfg.model.num_classes == 2:
    precision, recall, ap = compute_binary_pr(model, test_loader)
    plot_binary_pr(
        precision,
        recall,
        ap,
        class_names=class_name_set,
        save_path=os.path.join(fig_dir, "pr_curve.png")
    )
else:
    precision, recall, ap = compute_multiclass_pr(model, test_loader)
    plot_multiclass_pr(
        precision,
        recall,
        ap,
        class_names=class_name_set,
        save_path=os.path.join(fig_dir, "pr_curve.png")
    )

from evaluation.embedding import collect_latents, plot_umap

latents, labels = collect_latents(model, test_loader)
plot_umap(latents.numpy(), labels.numpy(), class_names=class_name_set, save_path=os.path.join(fig_dir, "umap.png"))

from evaluation.publish_figure import plot_publish_figure

plot_publish_figure(
    cm=cm,
    latent_vectors=latents.numpy(),
    latent_labels=labels.numpy(),
    precision=precision,
    recall=recall,
    ap=ap,
    fpr=fpr,
    tpr=tpr,
    roc_auc=roc_auc,
    class_names=class_name_set,
    normalize_cm=False,
    save_path=os.path.join(fig_dir, "publish_fig.png"),
)

from evaluation.cam import GradCAM1D, generate_cam_for_sample, plot_cam
# get one batch
batch = next(iter(test_loader))

# pick target layer
if cfg.model_type == "onebranch":
    target_layer = model.model.branch.resblocks[1].block[3] # for onebranch
else:
    target_layer = model.model.xanes_branch.resblocks[1].block[3]

# create CAM object
cam = GradCAM1D(model, target_layer)

# generate CAMs for whole batch
cams = cam.generate_from_batch(batch)

# visualize first sample
xanes = batch[0][0].numpy()   # spectrum
plot_cam(xanes, cams[0].numpy(), save_path=os.path.join(fig_dir, "CAM.png"))

cam.remove_hooks()