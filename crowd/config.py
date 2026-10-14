from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    """Read config.yml and turn weight paths into absolute paths."""
    path = Path(path) if path else ROOT / "config.yml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("density_weights", "detector_weights"):
        cfg[key] = str(ROOT / cfg[key])
    return cfg
