import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.lightning_classifier import SpectraClassifierModule
from data.spectra_datamodule import SpectraDataModule

from evaluation.full_evaluation import run_full_evaluation

import datetime
from utils.random_word import get_random_word_combo

import torch
torch.set_float32_matmul_precision("high")

import matplotlib
matplotlib.use("Agg")

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Hydra ouptut directory: {run_dir}")

    # Seeds model init + dataloader shuffling/workers; data.seed (train/val/test
    # split) is set independently in SpectraDataModule.
    pl.seed_everything(cfg.train_seed, workers=True)

    # --- W&B logger ---
    logger = None
    import wandb
    wandb.finish()
    if cfg.logging.use_wandb:
        ct = datetime.datetime.now()
        random_tag = get_random_word_combo()
        run_name = f'nedo_class_{ct.year}_{ct.month}_{ct.day}-{ct.hour}:{ct.minute}:{ct.second}_{random_tag}'
        logger = WandbLogger(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=run_name,
            tags=cfg.logging.tags,
            config=dict(cfg),
            reinit=True,
        )

    # --- datamodule ---
    datamodule = SpectraDataModule(cfg.data, run_dir=run_dir)

    # --- model ---
    model = SpectraClassifierModule(cfg)
    
    # --- checkpoint ---
    ckpt_dir = os.path.join(run_dir, "checkpoints")  # hydra run dir is current working dir
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_macro_f1",         # or "val_acc" if you prefer
        mode="max",                 # "min" for loss, "max" for accuracy
        save_top_k=1,               # save only the best model
        filename = f"spectra-{cfg.model_type}-bs{cfg.data.batch_size}-epoch{{epoch:02d}}-val_loss{{val_loss:.4f}}",
        dirpath=ckpt_dir    # folder to save checkpoints
    )  
    
    early_stop = EarlyStopping(
        monitor="val_macro_f1",
        patience=30,
        mode="max"
    )
    
    if logger is not None:
        logger.experiment.log({"checkpoint_dir": ckpt_dir})

    if cfg.scheduler.early_stop:
        callback_block = [checkpoint_callback, early_stop]
    else:
        callback_block = [checkpoint_callback]

    # --- trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callback_block,
    )

    trainer.fit(model, datamodule=datamodule)
    
    # best_model = SpectraClassifierModule.load_from_checkpoint(
    #     checkpoint_callback.best_model_path,
    #     weights_only=False,
    # )
    # best_model.eval()
    # best_model.to("cuda")
    
    # run_full_evaluation(best_model, datamodule, run_dir, cfg)
    
    return trainer.callback_metrics["val_macro_f1"].item()

if __name__ == "__main__":
    main()
