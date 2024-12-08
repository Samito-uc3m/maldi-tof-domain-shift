from pathlib import Path

import pandas as pd


def DRIAMS_bin_to_df(DRIAMS_ROOT: Path) -> pd.DataFrame:
    """
    Gets a Driams directory and returns a DataFrame with the following data:
    - species
    - bin 6000
    """

    # Get the path to the DRIAMS-C clean csv
    clean_csv_path: Path = DRIAMS_ROOT / "id" / "2018" / "2018_clean.csv"
    df = pd.read_csv(clean_csv_path)
    df = df[["species", "code"]]
    df["binned_6000"] = None

    # Get the path to the DRIAMS-C bin 6000
    bin_6000_folder_path: Path = DRIAMS_ROOT / "binned_6000" / "2018"
    bin_6000_path_ext: str = ".txt"

    # For each row in the DataFrame, get the bin 6000
    for index, row in df.iterrows():
        code = row["code"]
        bin_6000_path = bin_6000_folder_path / (code + bin_6000_path_ext)
        try:
            bin_6000 = pd.read_csv(bin_6000_path, sep="\s+")["binned_intensity"].values
            df.at[index, "binned_6000"] = bin_6000
        except FileNotFoundError:
            print(f"File {bin_6000_path} not found")
            df.at[index, "binned_6000"] = None

    # Filter by rows with bin 6000
    df = df[df["binned_6000"].notnull()]

    return df
