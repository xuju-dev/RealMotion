# flake8: noqa: E302,E501
from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

from .av2_dataset import Av2Dataset, collate_fn


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        dataset: dict = {},
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
        logger=None,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.dataset_cfg = dataset
        self.batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        self.logger = logger

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(
                data_root=self.data_root, split='train', logger=self.logger, **self.dataset_cfg
            )
            self.val_dataset = Av2Dataset(
                data_root=self.data_root, split='val', logger=self.logger, **self.dataset_cfg
            )
        else:
            self.test_dataset = Av2Dataset(
                data_root=self.data_root, split='test', logger=self.logger, **self.dataset_cfg
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )
