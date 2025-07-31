import lightning as L
import torch.utils.data
from torch.utils.data import DataLoader

from hympi_ml.data import DataSource, ModelDataSpec


class RawDataModule(L.LightningDataModule):
    """
    A data module that links a single ModelDataSpec to a train, validation, and test DataSource.

    Note: It is "raw" due to the fact that the dataloaders only load raw data that is not filtered or transformed
    as this is done on the model-side during training to run on the GPU most effectively.
    """

    def __init__(
        self,
        spec: ModelDataSpec,
        train_source: DataSource,
        val_source: DataSource,
        test_source: DataSource,
        batch_size: int,
        num_workers: int = 20,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.spec = spec
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        batches = RawModelBatches(self.train_source, self.spec, self.batch_size)
        return DataLoader(
            batches,
            shuffle=True,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        batches = RawModelBatches(self.val_source, self.spec, self.batch_size)
        return DataLoader(
            batches,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        batches = RawModelBatches(self.test_source, self.spec, self.batch_size)
        return DataLoader(
            batches,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class RawModelBatches(torch.utils.data.Dataset):
    """
    Combines a ModelDataSpec with a DataSource and loads raw batches of data.
    As a PyTorch dataset, it is fully capable of being indexed and used in a DataLoader.
    """

    def __init__(self, source: DataSource, spec: ModelDataSpec, batch_size: int):
        self.batch_size = batch_size

        self.source = source

        self.spec = spec

    def __len__(self):
        return int(self.source.sample_count / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size

        return self.spec.load_raw_slice(self.source, start, end)
