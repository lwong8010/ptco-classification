import os
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import pytorch_lightning as pl
from hydra.utils import to_absolute_path

class SpectraDataModule(pl.LightningDataModule):
    def __init__(self, cfg, run_dir=None):
        """
        cfg: Hydra config for data
        run_dir: optional, Hydra run directory to store splits
        """
        super().__init__()
        self.cfg = cfg

        self.set_data_direct = getattr(cfg, "set_data_direct", False)
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        self.seed = cfg.seed

        self.train_set = None
        self.val_set = None
        self.test_set = None

        if self.set_data_direct:
            self.train_dataset_path = self._resolve_dataset_path(cfg.train_dataset_path)
            self.val_dataset_path = self._resolve_dataset_path(cfg.val_dataset_path)
            self.test_dataset_path = self._resolve_dataset_path(cfg.test_dataset_path)
        else:
            self.dataset_path = self._resolve_dataset_path(cfg.dataset_path)
            self.train_frac = cfg.train_frac
            self.val_frac = cfg.val_frac
            self.test_frac = cfg.test_frac

        # Use Hydra run directory if provided
        if run_dir is None:
            run_dir = os.getcwd()
        self.split_file = os.path.join(run_dir, "splits.pkl")

    @staticmethod
    def _resolve_dataset_path(path_str):
        """
        Resolve a configured dataset path to an absolute path inside the
        project's datasets/ directory, accepting either a bare filename
        (e.g. "pt_xe_tb.pt") or a "datasets/..."-prefixed path for
        backwards compatibility. Rejects anything that would resolve
        outside of datasets/ (e.g. "../secret.pt" or an absolute path
        elsewhere), so dataset files always stay local to that folder.
        """
        datasets_dir = os.path.realpath(to_absolute_path("datasets"))

        relative = path_str.replace(os.sep, "/")
        if relative.startswith("datasets/"):
            relative = relative[len("datasets/"):]

        candidate = os.path.realpath(os.path.join(datasets_dir, relative))

        if os.path.commonpath([datasets_dir, candidate]) != datasets_dir:
            raise ValueError(
                f"Dataset path '{path_str}' must resolve inside the 'datasets/' "
                f"directory (resolved to {candidate})"
            )

        return candidate

    @staticmethod
    def _load_tensor_dataset(path):
        data = torch.load(path)

        # Support both (X, y) and (X1, X2, y)
        if len(data) == 2:
            X, y = data
            return TensorDataset(X, y)
        elif len(data) == 3:
            X1, X2, y = data
            return TensorDataset(X1, X2, y)
        else:
            raise ValueError("Dataset must be (X, y) or (X1, X2, y)")

    def setup(self, stage=None):
        if self.set_data_direct:
            self._load_direct_sets()
            self._save_direct_splits()

            print(
                f"Loaded train: {self.train_dataset_path}\n"
                f"Loaded val:   {self.val_dataset_path}\n"
                f"Loaded test:  {self.test_dataset_path}\n"
                f"  Train: {len(self.train_set)}\n"
                f"  Val:   {len(self.val_set)}\n"
                f"  Test:  {len(self.test_set)}\n"
                f"  Splits saved in: {self.split_file}"
            )
            return

        full_dataset = self._load_tensor_dataset(self.dataset_path)
        n_total = len(full_dataset)

        train_idx, val_idx, test_idx = self._create_and_save_splits(full_dataset, n_total)

        self.train_set = Subset(full_dataset, train_idx)
        self.val_set   = Subset(full_dataset, val_idx)
        self.test_set  = Subset(full_dataset, test_idx)

        print(
            f"Loaded {self.dataset_path}\n"
            f"  Train: {len(self.train_set)}\n"
            f"  Val:   {len(self.val_set)}\n"
            f"  Test:  {len(self.test_set)}\n"
            f"  Splits saved in: {self.split_file}"
    )

    def _load_direct_sets(self):
        self.train_set = self._load_tensor_dataset(self.train_dataset_path)
        self.val_set = self._load_tensor_dataset(self.val_dataset_path)
        self.test_set = self._load_tensor_dataset(self.test_dataset_path)

    def _save_direct_splits(self):
        # Direct mode has no index split, so record which files were used
        with open(self.split_file, "wb") as f:
            pickle.dump({
                "seed": self.seed,
                "set_data_direct": True,
                "train_dataset_path": self.train_dataset_path,
                "val_dataset_path": self.val_dataset_path,
                "test_dataset_path": self.test_dataset_path,
                "train_size": len(self.train_set),
                "val_size": len(self.val_set),
                "test_size": len(self.test_set),
            }, f)

    def _create_and_save_splits(self, full_dataset, n_total):
        # Compute sizes
        n_train = int(n_total * self.train_frac)
        n_val = int(n_total * self.val_frac)
        n_test = n_total - n_train - n_val

        # Shuffle indices with seed
        g = torch.Generator()
        g.manual_seed(self.seed)
        idx = torch.randperm(n_total, generator=g).tolist()

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        # Save splits to Hydra run directory
        with open(self.split_file, "wb") as f:
            pickle.dump({
                "seed": self.seed,
                "train": train_idx,
                "val": val_idx,
                "test": test_idx,
            }, f)

        return train_idx, val_idx, test_idx

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
    def load_splits_from_run(self, run_dir):
        # Direct mode: the run's config already names the train/val/test
        # files, so splits.pkl isn't needed (runs from before it was saved
        # in this mode don't have one)
        if self.set_data_direct:
            self._load_direct_sets()
            print(
                f"Loaded train: {self.train_dataset_path}\n"
                f"Loaded val:   {self.val_dataset_path}\n"
                f"Loaded test:  {self.test_dataset_path}\n"
                f"  Train: {len(self.train_set)}\n"
                f"  Val:   {len(self.val_set)}\n"
                f"  Test:  {len(self.test_set)}"
            )
            return

        split_file = os.path.join(run_dir, "splits.pkl")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"No splits.pkl found in {run_dir}")

        full_dataset = self._load_tensor_dataset(self.dataset_path)

        with open(split_file, "rb") as f:
            splits = pickle.load(f)

        train_idx = splits["train"]
        val_idx   = splits["val"]
        test_idx  = splits["test"]

        self.train_set = Subset(full_dataset, train_idx)
        self.val_set   = Subset(full_dataset, val_idx)
        self.test_set  = Subset(full_dataset, test_idx)

        print(
            f"Loaded dataset: {self.dataset_path}\n"
            f"Splits loaded from: {split_file}\n"
            f"  Train: {len(self.train_set)}\n"
            f"  Val:   {len(self.val_set)}\n"
            f"  Test:  {len(self.test_set)}"
        )

