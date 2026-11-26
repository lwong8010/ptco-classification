# Pt–Co nanoparticle arrangement classifier

Code for the spectra classifier in XAFS-Based Identification of Bimetallic Nanoparticle Structural Motifs Using Learned Spectral Classifiers (DOI: )

The models take X-ray absorption spectra (XANES and/or EXAFS at the Pt and Co edges) of
Pt–Co nanoparticles and predict the chemical arrangement of the particle as one of three
classes: **Random**, **Core-Shell** or **Ordered**.

## Repository layout

```
train.py                        Train one model, or a sweep with -m
single_test.py                  Evaluate a trained run and make figures
class_model.py                  Network definitions
models/lightning_classifier.py  Lightning module: model selection, training/validation steps, optimizer and LR schedule
data/spectra_datamodule.py      Loads a dataset, makes and saves the train/val/test split
evaluation/                     Metrics, UMAP embedding, Grad-CAM and figure helpers
utils/                          Run-name generator; supervised contrastive loss (not enabled in this code)
configs/                        Hydra configuration
datasets/                       Spectra datasets (.pt)
```

## Installation

Tested with Python 3.11 on Linux with CUDA 12.8.

```bash
conda create -n ptco-class python=3.11
conda activate ptco-class
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

For a different CUDA version, swap the index URL for the one listed at
[pytorch.org](https://pytorch.org/get-started/locally/). Training uses a GPU if one is
available. `single_test.py` needs a CUDA GPU.

## Data

The datasets are in `datasets/`. Dataset paths in the configs are resolved inside
`datasets/`, so run all commands from the repository root.

Each file is a tuple of tensors saved with `torch.save`:

| Model type | File contents | Shapes |
|---|---|---|
| one-branch | `(X, y)` | `X`: float32 `(N, L)` |
| two-branch (`*_tb.pt`) | `(X1, X2, y)` | `X1`, `X2`: float32 `(N, L)` |

`y` is int64 with labels `0 = Random`, `1 = Core-Shell`, `2 = Ordered`. Each spectrum has
256 points. Datasets that combine spectra concatenate them along the last axis in the order
shown below (‖ = concatenation).

| Data config | File | Input(s) | Length | Used with |
|---|---|---|---|---|
| `pt_xanes` | `pt_avg_xanes.pt` | Pt XANES | 256 | `onebranch` |
| `pt_exafs` | `pt_avg_exafs.pt` | Pt EXAFS | 256 | `onebranch` |
| `co_xanes` | `co_avg_xanes.pt` | Co XANES | 256 | `onebranch` |
| `co_exafs` | `co_avg_exafs.pt` | Co EXAFS | 256 | `onebranch` |
| `pt_xe` | `pt_avg_xanes_exafs.pt` | Pt XANES ‖ Pt EXAFS | 512 | `onebranch` |
| `co_xe` | `co_avg_xanes_exafs.pt` | Co XANES ‖ Co EXAFS | 512 | `onebranch` |
| `ptco_xx` | `ptco_avg_xanes.pt` | Pt XANES ‖ Co XANES | 512 | `onebranch` |
| `ptco_ee` | `ptco_avg_exafs.pt` | Pt EXAFS ‖ Co EXAFS | 512 | `onebranch` |
| `ptco_xxee` | `ptco_avg_xxee.pt` | Pt XANES ‖ Co XANES ‖ Pt EXAFS ‖ Co EXAFS | 1024 | `onebranch` |
| `ptco_xexe` | `ptco_avg_xexe.pt` | Pt XANES ‖ Pt EXAFS ‖ Co XANES ‖ Co EXAFS | 1024 | `onebranch` |
| `pt_xe_tb` | `pt_xe_tb.pt` | branch 1: Pt XANES · branch 2: Pt EXAFS | 256 / 256 | two-branch |
| `co_xe_tb` | `co_xe_tb.pt` | branch 1: Co XANES · branch 2: Co EXAFS | 256 / 256 | two-branch |
| `ptco_xx_tb` | `ptco_xanes_tb.pt` | branch 1: Pt XANES · branch 2: Co XANES | 256 / 256 | two-branch |
| `ptco_ee_tb` | `ptco_exafs_tb.pt` | branch 1: Pt EXAFS · branch 2: Co EXAFS | 256 / 256 | two-branch |
| `ptco_xxee_tb` | `ptco_xxee_tb.pt` | branch 1: Pt ‖ Co XANES · branch 2: Pt ‖ Co EXAFS | 512 / 512 | two-branch |
| `ptco_xexe_tb` | `ptco_xexe_tb.pt` | branch 1: Pt XANES ‖ EXAFS · branch 2: Co XANES ‖ EXAFS | 512 / 512 | two-branch |
| `ptx_coe` | `ptx_coe.pt` | Pt XANES ‖ Co EXAFS | 512 | `onebranch` |
| `ptx_coe_tb` | `ptx_coe_tb.pt` | branch 1: Pt XANES · branch 2: Co EXAFS | 256 / 256 | two-branch |

The Pt-containing datasets have 5,717 samples (1,329 / 2,622 / 1,766 per class). The Co-only
datasets have 5,759 (1,329 / 2,658 / 1,772).

## Models

Select with `model_type`:

| `model_type` | Description |
|---|---|
| `onebranch` | One spectral branch (1D CNN → two residual blocks → multi-head self-attention → adaptive pooling → MLP), then a classifier head. |
| `twobranch` | Two spectral branches, one per input; embeddings are concatenated and passed to the classifier head. |
| `twobranch_gf` | Two spectral branches combined by a learned sigmoid gate (gated fusion), then the classifier head. **Default.** |
| `twobranch_baseline` | Ablation: each branch is a single linear layer on the raw spectrum, followed by a linear classifier. |

All models are trained with cross-entropy loss using AdamW and a one-cycle cosine learning-rate
schedule. Training stops early if validation macro-F1 hasn't improved for 30 epochs (maximum
1,000 epochs). The checkpoint with the best validation macro-F1 is kept. Hyperparameters are in
`configs/model/default.yaml`, `configs/optim/adamw.yaml` and `configs/scheduler/onecycle_cos.yaml`.

## Training

```bash
# Default: twobranch_gf on ptco_xxee_tb
python train.py

# Any model type / dataset combination
python train.py model_type=twobranch data=pt_xe_tb

# A predefined model/dataset pair from configs/model_data/
python train.py +model_data=onebranch_ptco_xxee

# Full architecture × dataset comparison: 22 cases × 10 training seeds = 220 runs
python train.py -m
```

Each run is split 80/10/10 into train/val/test at random. The split is controlled by
`data.seed` (default 135), separately from `train_seed`, which sets model initialisation
and batch order. Every run with the same dataset therefore uses the same test set.

Runs are written to `outputs/<date>/<time>/` (`multirun/<date>/<time>/<n>/` for `-m`) and contain:

- `checkpoints/*.ckpt`: best checkpoint
- `.hydra/config.yaml`: the full resolved config
- `splits.pkl`: the train/val/test indices

### Weights & Biases

Logging to W&B is off by default. To turn it on:

```bash
python train.py logging.use_wandb=true logging.entity=your-wandb-entity
```

or edit `configs/logging/wandb.yaml`.

## Evaluation

Set `run_dir` at the top of `single_test.py` to a training run folder (e.g.
`outputs/2026-01-01/12-00-00`) and run:

```bash
python single_test.py
```

It loads the best checkpoint and that run's test split, then writes to `<run_dir>/figures/`:

| File | Contents |
|---|---|
| `test_metrics.json` | Accuracy, macro precision/recall/F1, micro-F1 |
| `confusion_matrix.png`, `confusion_matrix.npy` | Test-set confusion matrix |
| `roc_curve.png`, `pr_curve.png` | One-vs-rest ROC and precision–recall curves |
| `umap.png` | UMAP of the model's latent embeddings, coloured by class |
| `publish_fig.png` | Confusion matrix, UMAP, PR and ROC panels combined |
| `CAM.png` | Grad-CAM over the first branch's input for one test spectrum |
