import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from domain_shift.core.config import settings
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    driams_1 = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)
    driams_2 = DRIAMS_bin_to_df(settings.DRIAMS_D_PATH)

    # Filter by the 4 species with most representation
    most_represented_species = [
        "Escherichia coli",
        "Staphylococcus aureus",
        "Enterococcus faecalis",
        "Pseudomonas aeruginosa",
    ]
    driams_1 = driams_1[driams_1["species"].isin(most_represented_species)]
    driams_2 = driams_2[driams_2["species"].isin(most_represented_species)]

    # Get the binned data
    driams_1_binned_data = np.vstack(driams_1["binned_6000"].values)
    driams_2_binned_data = np.vstack(driams_2["binned_6000"].values)

    # Get a PCA of the data
    pca = PCA(n_components=3)
    pca.fit(driams_1_binned_data)
    pca_b = pca.transform(driams_1_binned_data)
    pca_c = pca.transform(driams_2_binned_data)

    # Mapping species to markers
    species_to_marker = {
        "Escherichia coli": "o",
        "Staphylococcus aureus": "^",
        "Enterococcus faecalis": "s",
        "Pseudomonas aeruginosa": "d",
    }

    # Plot PCA results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot DRIAMS 1 data
    for species, marker in species_to_marker.items():
        mask = driams_1["species"] == species
        ax.scatter(
            pca_b[mask, 0],
            pca_b[mask, 1],
            pca_b[mask, 2],
            c="r",
            marker=marker,
            label=f"DRIAMS C - {species}",
        )

    # Plot DRIAMS 2 data
    for species, marker in species_to_marker.items():
        mask = driams_2["species"] == species
        ax.scatter(
            pca_c[mask, 0],
            pca_c[mask, 1],
            pca_c[mask, 2],
            c="b",
            marker=marker,
            label=f"DRIAMS D - {species}",
        )

    # Add legend and labels
    ax.legend()
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # Move legend outside of the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.show()
