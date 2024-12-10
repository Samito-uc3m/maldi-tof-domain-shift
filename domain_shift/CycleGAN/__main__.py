from domain_shift.core.config import settings
from domain_shift.CycleGAN.data_loader import get_data_loader
from domain_shift.CycleGAN.models import CycleGAN
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    print("Loading data...")
    driams_1 = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)
    driams_2 = DRIAMS_bin_to_df(settings.DRIAMS_D_PATH)

    # Filter by the 4 species with most representation
    print("Filtering by the 4 species with most representation...")
    most_represented_species = [
        "Escherichia coli",
        "Staphylococcus aureus",
        "Enterococcus faecalis",
        "Pseudomonas aeruginosa",
    ]
    driams_1 = driams_1[driams_1["species"].isin(most_represented_species)]
    driams_2 = driams_2[driams_2["species"].isin(most_represented_species)]

    # Get the data_loaders
    print("Creating DataLoaders...")
    driams_1_data_loader = get_data_loader(driams_1)
    driams_2_data_loader = get_data_loader(driams_2)

    # Create the model
    print("Creating CycleGAN model...")
    model = CycleGAN()

    # Train the model
    print("Training CycleGAN model...")
    model.train(driams_1_data_loader, driams_2_data_loader)

    # Save the model
    print("Saving CycleGAN model...")
    model.save_models()
