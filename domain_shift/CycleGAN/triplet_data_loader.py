import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from domain_shift.core.config import settings


class BinnedTripletDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing at least:
              - 'binned_6000': a list/array of features for each sample
              - 'species': the class species or some identifier
        """
        self.data = dataframe.reset_index(
            drop=True
        )  # Re-index to ensure consecutive indices

        # Build a dictionary: species -> list of indices
        self.species_to_indices = {}
        for idx, row in self.data.iterrows():
            species = row["species"]
            if species not in self.species_to_indices:
                self.species_to_indices[species] = []
            self.species_to_indices[species].append(idx)

        # Store a list of all unique speciess
        self.speciess = list(self.species_to_indices.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a single triplet (anchor, positive, negative).

        - Anchor:  data from index `idx`
        - Positive: data from the same species as Anchor, different index if possible
        - Negative: data from a different species
        """
        # ----- Anchor -----
        anchor_row = self.data.iloc[idx]
        anchor_species = anchor_row["species"]
        anchor_binned = anchor_row["binned_6000"]
        anchor = torch.tensor(anchor_binned, dtype=torch.float32)

        # ----- Positive -----
        # All indices of the same species
        pos_candidates = self.species_to_indices[anchor_species]

        # We want a different index than 'idx'
        if len(pos_candidates) > 1:
            # If there's more than one candidate, pick a random one != idx
            positive_index = idx
            while positive_index == idx:
                positive_index = random.choice(pos_candidates)
        else:
            # Edge case: only one sample in this species => fallback
            positive_index = idx

        positive_binned = self.data.iloc[positive_index]["binned_6000"]
        positive = torch.tensor(positive_binned, dtype=torch.float32)

        # ----- Negative -----
        # Pick a species different from anchor_species
        neg_species_candidates = [lbl for lbl in self.speciess if lbl != anchor_species]
        negative_species = random.choice(neg_species_candidates)
        negative_index = random.choice(self.species_to_indices[negative_species])
        negative_binned = self.data.iloc[negative_index]["binned_6000"]
        negative = torch.tensor(negative_binned, dtype=torch.float32)

        return anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)


def get_data_loader(df: pd.DataFrame) -> DataLoader:
    """
    Returns a DataLoader instance for the given DataFrame.
    """
    dataset = BinnedTripletDataset(df)
    return DataLoader(
        dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=settings.BATCH_SHUFFLE,
        num_workers=settings.BATCH_NUM_WORKERS,
    )


class CosineLosslessTripletLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.epsilon = torch.finfo(torch.float32).eps  # A small value to avoid log(0)
        self.beta = beta  # The scaling parameter of the non-linear part

    def non_linearity(self, x):
        """
        Non-linear part of the loss function.

        Parameters:
        - x: Input tensor.

        Returns:
        - y: Output tensor.
        """
        antilog = -x / self.beta + 1
        antilog[antilog == 0] = self.epsilon
        return -torch.log(antilog)

    def forward(self, anchor, positive, negative):
        """
        Calculates Lossless Triplet Loss using cosine distance.

        Parameters:
        - anchor: Tensor of anchors.
        - positive: Tensor of positive examples.
        - negative: Tensor of negative examples.

        Returns:
        - loss: The loss value.
        """
        # Calculate cosine similarities
        pos_similarity = F.cosine_similarity(anchor, positive)
        neg_similarity = F.cosine_similarity(anchor, negative)

        # Convert similarities to normalized distances
        pos_dist = (1 - pos_similarity) / 2
        neg_dist = (1 - neg_similarity) / 2

        # Calculate non-linear parts of the loss function
        pos_loss = self.non_linearity(pos_dist)
        neg_loss = self.non_linearity(1.0 - neg_dist)

        loss = pos_loss + neg_loss
        return loss.mean()  # Return the average loss
