"""Local path configuration for the analysis notebooks.

Override any value with an environment variable, or edit these example defaults.
The training scripts also accept their input/output paths as command-line flags.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


DATA_ROOT = _configured_path("PROTOMRI_DATA_ROOT", PROJECT_ROOT / "data")
OUTPUT_ROOT = _configured_path("PROTOMRI_OUTPUT_ROOT", PROJECT_ROOT / "runs")
NSD_ROOT = _configured_path("PROTOMRI_NSD_ROOT", DATA_ROOT / "NSD_dataset")
COCO_IMAGES = _configured_path("PROTOMRI_COCO_IMAGES", DATA_ROOT / "mscoco" / "val2014")
PROTOTYPE_RUN = _configured_path("PROTOMRI_PROTOTYPE_RUN", OUTPUT_ROOT / "prototypes")
FMRI_RUNS = _configured_path("PROTOMRI_FMRI_RUNS", OUTPUT_ROOT / "fmri")
FMRI_MULTI_RUNS = _configured_path("PROTOMRI_FMRI_MULTI_RUNS", OUTPUT_ROOT / "fmri_multi")
