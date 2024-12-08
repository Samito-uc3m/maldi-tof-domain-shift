import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from domain_shift.core.config import settings
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    driams_b = DRIAMS_bin_to_df(settings.DRIAMS_B_PATH)
    driams_c = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)

    # Filter by the species with mora than 5 samples in b and c (intersection)
    b_species_counts = driams_b["species"].value_counts()
    c_species_counts = driams_c["species"].value_counts()
    b_species = b_species_counts[b_species_counts > 5].index
    c_species = c_species_counts[c_species_counts > 5].index
    selected_species = b_species.intersection(c_species)
    driams_b = driams_b[driams_b["species"].isin(selected_species)]
    driams_c = driams_c[driams_c["species"].isin(selected_species)]

    # Map the species to integers
    species_map = {species: i for i, species in enumerate(selected_species)}
    driams_b["species"] = driams_b["species"].map(species_map)
    driams_c["species"] = driams_c["species"].map(species_map)

    # Get the binned data
    b_binned_data = np.vstack(driams_b["binned_6000"].values)
    c_binned_data = np.vstack(driams_c["binned_6000"].values)

    # Create the labels
    b_labels = driams_b["species"].values
    c_labels = driams_c["species"].values

    # Train, valid and test with b_data and test with c_data
    b_train_data, b_test_data, b_train_labels, b_test_labels = train_test_split(
        b_binned_data, b_labels, test_size=0.2, random_state=42
    )

    # Train the classifier
    clf = RandomForestClassifier()
    clf.fit(b_train_data, b_train_labels)

    # Predict the driams b data
    b_test_predictions = clf.predict(b_test_data)
    b_test_f1 = f1_score(b_test_labels, b_test_predictions, average="weighted")
    print(f"Test f1 with b data: {b_test_f1}")

    # Predict the draims c data
    c_predictions = clf.predict(c_binned_data)
    c_f1 = f1_score(c_labels, c_predictions, average="weighted")
    print(f"f1 with c data: {c_f1}")
