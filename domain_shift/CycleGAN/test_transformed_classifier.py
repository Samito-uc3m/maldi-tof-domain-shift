import numpy as np
import torch
from sklearn.metrics import f1_score

from domain_shift.core.config import settings
from domain_shift.CycleGAN.data_loader import get_data_loader
from domain_shift.CycleGAN.models import Generator
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

print("Loading data...")
driams = DRIAMS_bin_to_df(settings.DRIAMS_D_PATH)

# Filter by the 4 species with most representation
print("Filtering by the 4 species with most representation...")
most_represented_species = [
    "Escherichia coli",
    "Staphylococcus aureus",
    "Enterococcus faecalis",
    "Pseudomonas aeruginosa",
]
driams = driams[driams["species"].isin(most_represented_species)]

# Map the species to integers
species_map = {species: i for i, species in enumerate(most_represented_species)}
driams["species"] = driams["species"].map(species_map)

# Get the data_loaders
print("Creating DataLoaders...")
driams_data_loader = get_data_loader(driams)

# Load the model
print("Loading Generator model...")
model = Generator()
model.load_state_dict(torch.load(settings.GENERATOR_2_TO_1_PATH))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Transform the data
print("Transforming data...")
transformed_data = []
for data in driams_data_loader:
    output = model(data.to(device)).cpu().detach()
    transformed_data.extend(output.cpu().detach().flatten(1))
transformed_data = torch.stack(transformed_data)
print(transformed_data.shape)

# Load classifier
print("Loading classifier...")
clf = torch.load(settings.CLASSIFIER_PATH)

# Predict the transformed data
print("Predicting transformed data...")
predictions = clf.predict(transformed_data.numpy())
transformed_data_labels = driams["species"].values
f1 = f1_score(transformed_data_labels, predictions, average="weighted")
print(f"Weighted f1: {f1}")

# Predict the original data
print("Predicting original data...")
predictions = clf.predict(np.vstack(driams["binned_6000"].values))
original_data_labels = driams["species"].values
f1 = f1_score(original_data_labels, predictions, average="weighted")
print(f"Weighted f1: {f1}")
