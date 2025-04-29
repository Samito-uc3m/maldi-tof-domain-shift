import matplotlib.pyplot as plt
import torch

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

# Get one random sample
print("Getting one random sample...")
driams = driams.sample(1)

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

# Show the original and transformed data

original_data = driams_data_loader.dataset[0]
original_data = original_data.numpy()
transformed_data = transformed_data.cpu().detach().numpy()

plt.figure()
plt.plot(original_data[0], label="Original data")
plt.legend()
plt.savefig(settings.IMAGES_PATH / "original_maldi-tof.png")

plt.figure()
plt.plot(transformed_data[0], label="Transformed data")
plt.legend()
plt.savefig(settings.IMAGES_PATH / "transformed_maldi-tof.png")
