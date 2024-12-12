from pathlib import Path

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
    GENERATOR_1_TO_2_PATH: Path = MODEL_PATH / "generator_C_to_D.pth"
    GENERATOR_2_TO_1_PATH: Path = MODEL_PATH / "generator_D_to_C.pth"
    DISCRIMINATOR_1_PATH: Path = MODEL_PATH / "discriminator_C.pth"
    DISCRIMINATOR_2_PATH: Path = MODEL_PATH / "discriminator_D.pth"
    CLASSIFIER_PATH: Path = MODEL_PATH / "classifier_C.pth"

    # Batch parameters
    BATCH_SIZE: int = 4
    BATCH_SHUFFLE: bool = True
    BATCH_NUM_WORKERS: int = 2

    # Training parameters
    EPOCHS: int = 10
    LR: float = 0.0002

    # Label values
    REAL_LABEL: float = 1.0
    FAKE_LABEL: float = 0.0


settings = Settings()
