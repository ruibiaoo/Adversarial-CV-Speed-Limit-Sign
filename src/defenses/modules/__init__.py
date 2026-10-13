# defenses/modules also exposes modelling/modules (model, data, augmentations),
# so `from modules.model import ...` keeps working without editing modelling.
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "modelling" / "modules"))
