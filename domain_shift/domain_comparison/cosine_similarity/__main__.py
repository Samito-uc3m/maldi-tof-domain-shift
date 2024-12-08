import numpy as np

from domain_shift.core.config import settings
from domain_shift.data_extraction.process_DRIAMS import DRIAMS_bin_to_df

if __name__ == "__main__":
    driams_b = DRIAMS_bin_to_df(settings.DRIAMS_B_PATH)
    driams_c = DRIAMS_bin_to_df(settings.DRIAMS_C_PATH)

    # Filter by the 1 species with most representation
    most_represented_species = ["Escherichia coli"]
    driams_b = driams_b[driams_b["species"].isin(most_represented_species)]
    driams_c = driams_c[driams_c["species"].isin(most_represented_species)]

    # Get the binned data
    b_binned_data = np.vstack(driams_b["binned_6000"].values)
    c_binned_data = np.vstack(driams_c["binned_6000"].values)

    b_avg_similarity = 0
    c_avg_similarity = 0
    for index in range(len(b_binned_data)):
        # Get the similarity between the binned data
        b_data = b_binned_data[index]

        for index2 in range(len(b_binned_data)):
            b2_data = b_binned_data[index2]
            similarity = np.dot(b_data, b2_data) / (
                np.linalg.norm(b_data) * np.linalg.norm(b2_data)
            )
            b_avg_similarity += similarity
        b_avg_similarity /= len(b_binned_data)

        for index2 in range(len(c_binned_data)):
            c_data = c_binned_data[index2]
            similarity = np.dot(b_data, c_data) / (
                np.linalg.norm(b_data) * np.linalg.norm(c_data)
            )
            c_avg_similarity += similarity
        c_avg_similarity /= len(c_binned_data)

    b_avg_similarity /= len(b_binned_data)
    c_avg_similarity /= len(b_binned_data)
    print(f"Average similarity between b binned data: {b_avg_similarity}")
    print(f"Average similarity between c binned data: {c_avg_similarity}")
