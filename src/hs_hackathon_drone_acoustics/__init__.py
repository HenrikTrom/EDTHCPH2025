from pathlib import Path
from typing import Literal

CLASS_TYPE = Literal["background", "drone", "helicopter"]
CLASSES: list[CLASS_TYPE] = ["background", "drone", "helicopter"]

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "examples"
print(EXAMPLES_DIR)
