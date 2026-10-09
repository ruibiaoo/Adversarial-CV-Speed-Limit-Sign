# Phase_3/modules also exposes Phase_1/modules (model, data, augmentations),
# so `from modules.model import ...` keeps working without editing Phase 1.
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "Phase_1" / "modules"))
