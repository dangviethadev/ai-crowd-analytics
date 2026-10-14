from pathlib import Path

from crowd.config import ROOT, load_config


def test_root_is_project_folder():
    assert (ROOT / "config.yml").exists()


def test_load_config_resolves_weight_paths():
    cfg = load_config()
    assert Path(cfg["density_weights"]).is_absolute()
    assert Path(cfg["detector_weights"]).is_absolute()  # model files are not in git
    assert cfg["level_thresholds"]["low"] < cfg["level_thresholds"]["high"]
