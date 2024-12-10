import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from domain_shift.core.config import settings


class BinndeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'binned_6000'.
        """
        self.data = dataframe

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a single sample from the dataset at index `idx`.
        """
        # Extract features for the given index
        # Since we are talking about GAN's, we don't need labels
        features = torch.tensor(self.data.iloc[idx]["binned_6000"], dtype=torch.float32)
        return features.unsqueeze(0)


def get_data_loader(df: pd.DataFrame) -> DataLoader:
    """
    Returns a DataLoader instance for the given DataFrame.
    """
    dataset = BinndeDataset(df)
    return DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=settings.BATCH_SHUFFLE,
        num_workers=settings.BATCH_NUM_WORKERS,
    )
