from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DRIAMS_B_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/data/DRIAMS-B"
    )
    DRIAMS_C_PATH: Path = Path(
        "/Users/sam/Documents/Universidad/Master/TFM/maldi-tof-domain-shift/data/DRIAMS-C"
    )


settings = Settings()
