from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Data paths
    DRIAMS_B_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/data/DRIAMS-B"
    )
    DRIAMS_C_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/data/DRIAMS-C"
    )
    DRIAMS_D_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/data/DRIAMS-D"
    )

    # Model saving/loading paths
    MODEL_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/models"
    )
    GENERATOR_1_TO_2_PATH: Path = MODEL_PATH / "generator_1_to_2.pth"
    GENERATOR_2_TO_1_PATH: Path = MODEL_PATH / "generator_2_to_1.pth"
    DISCRIMINATOR_1_PATH: Path = MODEL_PATH / "discriminator_1.pth"
    DISCRIMINATOR_2_PATH: Path = MODEL_PATH / "discriminator_2.pth"

    # Batch parameters
    BATCH_SIZE: int = 16
    BATCH_SHUFFLE: bool = True
    BATCH_NUM_WORKERS: int = 2

    # Training parameters
    EPOCHS: int = 100
    LR: float = 0.0002

    # Label values
    REAL_LABEL: float = 1.0
    FAKE_LABEL: float = 0.0


settings = Settings()
