import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from domain_shift.core.config import settings


class BinnedTripletDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, augment: bool = True) -> None:
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing at least:
              - 'binned_6000': a list/array of features for each sample
              - 'species': the class species or some identifier
        """
        self.data = dataframe.reset_index(drop=True)

        # Convert 'binned_6000' to NumPy arrays for fast distance computations
        self.data["binned_6000"] = self.data["binned_6000"].apply(lambda x: np.array(x))

        # Augment data
        if augment:
            self.augment_data()

        # Build a dictionary: species -> list of indices
        self.species_to_indices = {}
        for idx, row in self.data.iterrows():
            species = row["species"]
            if species not in self.species_to_indices:
                self.species_to_indices[species] = []
            self.species_to_indices[species].append(idx)

        # Store a list of all unique species
        self.species_list = list(self.species_to_indices.keys())

    def augment_data(self):
        """Creates augmented data by shifting binned_6000 left and right by one position."""
        augmented_rows = []
        for _, row in self.data.iterrows():
            binned = row["binned_6000"]
            species = row["species"]

            if len(binned) > 1:
                shifted_left = np.roll(binned, -1)
                shifted_right = np.roll(binned, 1)

                augmented_rows.append({"binned_6000": shifted_left, "species": species})
                augmented_rows.append(
                    {"binned_6000": shifted_right, "species": species}
                )

        self.data = pd.concat(
            [self.data, pd.DataFrame(augmented_rows)], ignore_index=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns a triplet (anchor, positive, negative):
        - Anchor: Sample at index `idx`
        - Positive: Furthest sample of the same species (max distance)
        - Negative: Closest sample from another species (min distance)
        """
        # ----- Anchor -----
        anchor_row = self.data.iloc[idx]
        anchor_species = anchor_row["species"]
        anchor_binned = anchor_row["binned_6000"]
        anchor = torch.tensor(anchor_binned, dtype=torch.float32)

        # ----- Positive (Furthest of Same Species) -----
        pos_candidates = self.species_to_indices[anchor_species]
        if len(pos_candidates) > 1:
            # Compute distances to all candidates of the same species
            dists = [
                (i, np.linalg.norm(anchor_binned - self.data.iloc[i]["binned_6000"]))
                for i in pos_candidates
                if i != idx
            ]
            positive_index = max(dists, key=lambda x: x[1])[0]  # Max distance
        else:
            positive_index = idx  # Edge case: fallback to self

        positive_binned = self.data.iloc[positive_index]["binned_6000"]
        positive = torch.tensor(positive_binned, dtype=torch.float32)

        # ----- Negative (Closest from Another Species) -----
        neg_candidates = [
            i
            for i in range(len(self.data))
            if self.data.iloc[i]["species"] != anchor_species
        ]

        if neg_candidates:
            dists = [
                (i, np.linalg.norm(anchor_binned - self.data.iloc[i]["binned_6000"]))
                for i in neg_candidates
            ]
            negative_index = min(dists, key=lambda x: x[1])[0]  # Min distance
        else:
            negative_index = random.choice(range(len(self.data)))  # Fallback case

        negative_binned = self.data.iloc[negative_index]["binned_6000"]
        negative = torch.tensor(negative_binned, dtype=torch.float32)

        return anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)


def get_data_loader(df: pd.DataFrame, augment: bool = True) -> DataLoader:
    """
    Returns a DataLoader instance for the given DataFrame.
    """
    dataset = BinnedTripletDataset(df, augment)
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
