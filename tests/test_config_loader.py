from riceleaf.config.loader import load_config


def test_load_default():
    cfg = load_config("configs/experiments/E0_baseline_cnn.yml")
    assert cfg.experiment.id == "E0"
