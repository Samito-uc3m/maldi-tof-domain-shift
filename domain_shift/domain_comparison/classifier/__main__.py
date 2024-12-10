import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from domain_shift.core.config import settings
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    driams_1 = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)
    driams_2 = DRIAMS_bin_to_df(settings.DRIAMS_D_PATH)

    # Filter by the species with mora than 5 samples in b and c (intersection)
    driams_1_species_counts = driams_1["species"].value_counts()
    driams_2_species_counts = driams_2["species"].value_counts()
    driams_1_species = driams_1_species_counts[driams_1_species_counts > 5].index
    driams_2_species = driams_2_species_counts[driams_2_species_counts > 5].index
    selected_species = driams_1_species.intersection(driams_2_species)
    driams_1 = driams_1[driams_1["species"].isin(selected_species)]
    driams_2 = driams_2[driams_2["species"].isin(selected_species)]

    # Map the species to integers
    species_map = {species: i for i, species in enumerate(selected_species)}
    driams_1["species"] = driams_1["species"].map(species_map)
    driams_2["species"] = driams_2["species"].map(species_map)

    # Get the binned data
    driams_1_binned_data = np.vstack(driams_1["binned_6000"].values)
    driams_2_binned_data = np.vstack(driams_2["binned_6000"].values)

    # Create the labels
    driams_1_labels = driams_1["species"].values
    driams_2_labels = driams_2["species"].values

    # Train, valid and test with driams_1_data and test with driams_2_data
    (
        driams_1_train_data,
        driams_1_test_data,
        driams_1_train_labels,
        driams_1_test_labels,
    ) = train_test_split(
        driams_1_binned_data, driams_1_labels, test_size=0.2, random_state=42
    )

    # Train the classifier
    clf = RandomForestClassifier()
    clf.fit(driams_1_train_data, driams_1_train_labels)

    # Predict the driams b data
    driams_1_test_predictions = clf.predict(driams_1_test_data)
    driams_1_test_f1 = f1_score(
        driams_1_test_labels, driams_1_test_predictions, average="weighted"
    )
    print(f"Test f1 with driams 1 data: {driams_1_test_f1}")

    # Predict the draims c data
    driams_2_predictions = clf.predict(driams_2_binned_data)
    driams_2_f1 = f1_score(driams_2_labels, driams_2_predictions, average="weighted")
    print(f"f1 with driams 2 data: {driams_2_f1}")
