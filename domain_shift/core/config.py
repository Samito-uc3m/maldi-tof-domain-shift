from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Data paths
    DRIAMS_B_PATH: Path = Path(
        "/home/dive001/Documents/Master/maldi-tof-domain-shift/data/DRIAMS-B"
    )
    DRIAMS_C_PATH: Path = Path(
        "/home/dive001/Documents/Master/maldi-tof-domain-shift/data/DRIAMS-C"
    )
    DRIAMS_D_PATH: Path = Path(
        "/home/dive001/Documents/Master/maldi-tof-domain-shift/data/DRIAMS-D"
    )

    # Images folder
    IMAGES_PATH: Path = Path(
        "/home/dive001/Documents/Master/maldi-tof-domain-shift/images"
    )

    # Model saving/loading paths
    MODEL_PATH: Path = Path(
        "/home/dive001/Documents/Master/maldi-tof-domain-shift/models"
    )
    CYCLEGAN_FOLDER_PATH: Path = MODEL_PATH / "Escherichia_coli"
    GENERATOR_1_TO_2_PATH: Path = CYCLEGAN_FOLDER_PATH / "generator_C_to_D.pth"
    GENERATOR_2_TO_1_PATH: Path = CYCLEGAN_FOLDER_PATH / "generator_D_to_C.pth"
    DISCRIMINATOR_1_PATH: Path = CYCLEGAN_FOLDER_PATH / "discriminator_C.pth"
    DISCRIMINATOR_2_PATH: Path = CYCLEGAN_FOLDER_PATH / "discriminator_D.pth"
    CLASSIFIER_PATH: Path = MODEL_PATH / "classifier_C.pth"

    # Batch parameters
    BATCH_SIZE: int = 3
    BATCH_SHUFFLE: bool = True
    BATCH_NUM_WORKERS: int = 2

    # Training parameters
    EPOCHS: int = 20
    LR: float = 0.0002

    # Label values
    REAL_LABEL: float = 1.0
    FAKE_LABEL: float = 0.0

    # Make sure the model folders exist
    @model_validator(mode="after")
    def _check_model_folders(self):
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.CYCLEGAN_FOLDER_PATH.mkdir(parents=True, exist_ok=True)


settings = Settings()
