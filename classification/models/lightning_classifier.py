from omegaconf import OmegaConf
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import F1Score
from class_model import (
    TwoBranchSpectraClassifier,
    TwoBranchSpectraClassifierGF,
    OneBranchSpectraClassifier,
    TwoBranchSpectraClassifierBaseline,
)
from utils.special_loss import (
    SupervisedContrastiveLoss,
    CombinedLoss,
)

class SpectraClassifierModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)  # logs everything automatically
        
        self.val_f1 = F1Score(
            task="multiclass",
            num_classes=cfg.model.num_classes,
            average="macro"
        )

        # --- model ---
        # -----------------------------
        # Instantiate model based on cfg.model.type
        # -----------------------------
        model_type = cfg.model_type.lower()
        if model_type == "twobranch":
            self.model = TwoBranchSpectraClassifier(
                xanes_length=cfg.model.xanes_length,
                exafs_length=cfg.model.exafs_length,
                num_classes=cfg.model.num_classes,
                branch_channels=cfg.model.branch_channels,
                branch_hidden=cfg.model.branch_hidden,
                head_hidden=cfg.model.head_hidden,
                dropout=cfg.model.dropout,
            )
            self.is_two_branch = True
            self.use_contrastive = False
        elif model_type == "twobranch_gf":
            self.model = TwoBranchSpectraClassifierGF(
                xanes_length=cfg.model.xanes_length,
                exafs_length=cfg.model.exafs_length,
                num_classes=cfg.model.num_classes,
                branch_channels=cfg.model.branch_channels,
                branch_hidden=cfg.model.branch_hidden,
                head_hidden=cfg.model.head_hidden,
                dropout=cfg.model.dropout,
                proj_hidden=cfg.model.get("proj_hidden", 128),
                proj_out=cfg.model.get("proj_out", 64),
            )
            self.is_two_branch = True
            self.use_contrastive = False
        elif model_type == "twobranch_baseline":
            self.model = TwoBranchSpectraClassifierBaseline(
                xanes_length=cfg.model.xanes_length,
                exafs_length=cfg.model.exafs_length,
                num_classes=cfg.model.num_classes,
                branch_channels=cfg.model.branch_channels,
                branch_hidden=cfg.model.branch_hidden,
                head_hidden=cfg.model.head_hidden,
                dropout=cfg.model.dropout,
            )
            self.is_two_branch = True
            self.use_contrastive = False
        elif model_type == "onebranch":
            self.model = OneBranchSpectraClassifier(
                input_length=cfg.model.input_length,
                num_classes=cfg.model.num_classes,
                branch_channels=cfg.model.branch_channels,
                branch_hidden=cfg.model.branch_hidden,
                head_hidden=cfg.model.head_hidden,
                dropout=cfg.model.dropout,
            )
            self.is_two_branch = False
            self.use_contrastive = False
        else:
            raise ValueError(f"Unknown model type: {cfg.model_type}")

        # --- loss ---
        if self.use_contrastive:
            # Optional: up-weight random <-> layered confusion
            # Classes: 0=random, 1=layered, 2=core-shell
            pair_weights = None
            # In SpectraClassifierModule.__init__, replace the pair_weights block:
            if getattr(cfg.loss, "use_pair_weights", False):
                pw_rl = cfg.loss.get("pw_random_to_layered", 2.0)  # random anchor → layered negative
                pw_lr = cfg.loss.get("pw_layered_to_random", 2.0)  # layered anchor → random negative
                pair_weights = torch.tensor([
                    [1.0, pw_rl, 1.0],
                    [pw_lr, 1.0, 1.0],
                    [1.0,  1.0,  1.0],
                ], dtype=torch.float32)

            self.criterion = CombinedLoss(
                num_classes=cfg.model.num_classes,
                temperature=cfg.loss.get("temperature", 0.1),
                lam=0.0,   # start at 0; annealing will ramp this up
                class_weights=None,
                pair_weights=pair_weights,
            )
            # Annealing config
            self._lam_max    = cfg.loss.get("lam_max", 0.3)
            self._lam_warmup = cfg.loss.get("lam_warmup_epochs", 10)
        else:
            self.criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, xanes, exafs=None, return_projection=False):
        if self.is_two_branch:
            if isinstance(self.model, TwoBranchSpectraClassifierGF):
                return self.model(xanes, exafs, return_projection=return_projection)
            return self.model(xanes, exafs)
        else:
            return self.model(xanes)

    # -----------------------------
    # λ Annealing
    # -----------------------------
    def on_train_epoch_start(self):
        if not self.use_contrastive:
            return
        # Linear ramp: 0 → lam_max over lam_warmup_epochs, then hold
        epoch = self.current_epoch
        new_lam = min(self._lam_max, self._lam_max * (epoch / max(self._lam_warmup, 1)))
        self.criterion.lam = new_lam
        self.log("contrastive_lambda", new_lam, prog_bar=True, on_epoch=True)

    # -----------------------------
    # Training Step
    # -----------------------------
    def training_step(self, batch, batch_idx):
        if self.is_two_branch:
            xanes, exafs, labels = batch
            xanes = xanes.unsqueeze(1)
            exafs = exafs.unsqueeze(1)

            if self.use_contrastive:
                logits, projections = self(xanes, exafs, return_projection=True)
                loss, ce_loss, con_loss = self.criterion(logits, projections, labels)
                self.log("train_ce_loss",  ce_loss,  prog_bar=False, on_step=False, on_epoch=True)
                self.log("train_con_loss", con_loss, prog_bar=False, on_step=False, on_epoch=True)
            else:
                logits = self(xanes, exafs)
                loss = self.criterion(logits, labels)

        else:
            spectra, labels = batch
            spectra = spectra.unsqueeze(1)
            logits = self(spectra)
            loss = self.criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # -----------------------------
    # Validation Step
    # -----------------------------
    def validation_step(self, batch, batch_idx):
        if self.is_two_branch:
            xanes, exafs, labels = batch
            xanes = xanes.unsqueeze(1)
            exafs = exafs.unsqueeze(1)
            logits = self(xanes, exafs)   # no projection at val time
        else:
            spectra, labels = batch
            spectra = spectra.unsqueeze(1)
            logits = self(spectra)

        # Val always uses plain CE regardless of contrastive setup
        ce_loss = (
            self.criterion.ce(logits, labels)
            if self.use_contrastive
            else self.criterion(logits, labels)
        )

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.val_f1.update(preds, labels)

        self.log("val_loss",     ce_loss,                        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_acc",      acc,                            prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_macro_f1", self.val_f1(preds, labels),     prog_bar=True,  on_step=False, on_epoch=True)

    # -----------------------------
    # Optimizer + Scheduler
    # -----------------------------
    def configure_optimizers(self):
        cfg = self.hparams

        # --- optimizer ---
        if cfg.optim.name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
        elif cfg.optim.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
        elif cfg.optim.name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optim.name}")

        # --- scheduler ---
        sched_cfg = cfg.scheduler
        if sched_cfg.name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_cfg.T_max,
                eta_min=sched_cfg.eta_min,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        elif sched_cfg.name.lower() == "cosine_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_cfg.T_0,  # usually T_max equivalent
                T_mult=getattr(sched_cfg, "T_mult", 1),
                eta_min=getattr(sched_cfg, "eta_min", 0),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        elif sched_cfg.name.lower() == "cyclic_tri2":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=sched_cfg.base_lr,
                max_lr=sched_cfg.max_lr,
                step_size_up=sched_cfg.step_size_up,
                mode="triangular2",
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif sched_cfg.name.lower() == "cyclic_exp":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=sched_cfg.base_lr,
                max_lr=sched_cfg.max_lr,
                step_size_up=sched_cfg.step_size_up,
                mode="exp_range",
                gamma=sched_cfg.exp_gamma,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif sched_cfg.name.lower() == "onecycle_cos":
            # Lightning expects steps per epoch for OneCycleLR
            if getattr(cfg.scheduler, "steps_per_epoch", None) is None:
                # datamodule must be passed to trainer.fit() to use this
                # fallback: estimate from length of train_dataloader
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                steps_per_epoch = cfg.scheduler.steps_per_epoch
            if steps_per_epoch is None:
                raise ValueError("OneCycleLR requires 'steps_per_epoch' in scheduler config")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=sched_cfg.max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=cfg.trainer.max_epochs,
                anneal_strategy="cos",
                pct_start=getattr(sched_cfg, "pct_start", 0.3),
                div_factor=getattr(sched_cfg, "div_factor", 25.0),
                final_div_factor=getattr(sched_cfg, "final_div_factor", 1e4),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        elif sched_cfg.name.lower() == "none":
            return optimizer

        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")
        
    def unpack_batch(self, batch):
        """
        Standardize batch handling for one-branch and two-branch models.
        Returns: inputs_dict, labels
        """
        if len(batch) == 2:
            xanes, labels = batch
            return {"xanes": xanes}, labels

        elif len(batch) == 3:
            xanes, exafs, labels = batch
            return {"xanes": xanes, "exafs": exafs}, labels

        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        
    def predict_step(self, batch, batch_idx):
        inputs, labels = self.unpack_batch(batch)

        # add channel dim here, once
        if "xanes" in inputs:
            inputs["xanes"] = inputs["xanes"].unsqueeze(1)
        if "exafs" in inputs:
            inputs["exafs"] = inputs["exafs"].unsqueeze(1)

        logits = self(**inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        return {
            "logits": logits.detach(),
            "probs": probs.detach(),
            "preds": preds.detach(),
            "labels": labels.detach(),
        }
        
    def encode(self, xanes=None, exafs=None):
        """
        Unified latent extractor supporting both OneBranch and TwoBranch models.
        """
        # Two-branch model
        if xanes is not None and exafs is not None:
            return self.model.encode(xanes, exafs)

        # One-branch model (XANES-only or EXAFS-only)
        if xanes is not None:
            return self.model.encode(xanes)

        if exafs is not None:
            return self.model.encode(exafs)

        raise ValueError("encode() called with no inputs")

    def extract_latents(self, batch):
        inputs, labels = self.unpack_batch(batch)

        if "xanes" in inputs:
            inputs["xanes"] = inputs["xanes"].unsqueeze(1)
        if "exafs" in inputs:
            inputs["exafs"] = inputs["exafs"].unsqueeze(1)

        z = self.encode(**inputs)
        return z.detach(), labels.detach()
    

