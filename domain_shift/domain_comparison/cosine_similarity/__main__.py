import numpy as np

from domain_shift.core.config import settings
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    driams_1 = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)
    driams_2 = DRIAMS_bin_to_df(settings.DRIAMS_D_PATH)

    # Filter by the 4 species with most representation
    most_represented_species = ["Escherichia coli"]
    driams_1 = driams_1[driams_1["species"].isin(most_represented_species)]
    driams_2 = driams_2[driams_2["species"].isin(most_represented_species)]

    # Get the binned data
    driams_1_binned_data = np.vstack(driams_1["binned_6000"].values)
    driams_2_binned_data = np.vstack(driams_2["binned_6000"].values)

    driams_1_avg_similarity = 0
    driams_2_avg_similarity = 0
    for index in range(len(driams_1_binned_data)):
        # Get the similarity between the binned data
        driams_1_data = driams_1_binned_data[index]

        for index2 in range(len(driams_1_binned_data)):
            b2_data = driams_1_binned_data[index2]
            similarity = np.dot(driams_1_data, b2_data) / (
                np.linalg.norm(driams_1_data) * np.linalg.norm(b2_data)
            )
            driams_1_avg_similarity += similarity
        driams_1_avg_similarity /= len(driams_1_binned_data)

        for index2 in range(len(driams_2_binned_data)):
            driams_2_data = driams_2_binned_data[index2]
            similarity = np.dot(driams_1_data, driams_2_data) / (
                np.linalg.norm(driams_1_data) * np.linalg.norm(driams_2_data)
            )
            driams_2_avg_similarity += similarity
        driams_2_avg_similarity /= len(driams_2_binned_data)

    driams_1_avg_similarity /= len(driams_1_binned_data)
    driams_2_avg_similarity /= len(driams_1_binned_data)
    print(f"Average similarity between Driams 1 binned data: {driams_1_avg_similarity}")
    print(f"Average similarity between Driams 2 binned data: {driams_2_avg_similarity}")
