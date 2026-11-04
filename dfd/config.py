"""Load standalone experiment YAML files without silently filling research choices."""

from pathlib import Path

import yaml


VARIANTS = {
    "teacher", "ti_ce", "ti_kd", "ti_factorized", "ti_dfd", "vietocr", "crnn",
    "no_tone_weight", "no_validity_mask", "joint_kd", "marginal_joint_kd",
    "tone_only_kd", "temperature_sweep",
}
UNSET_FIELDS = (
    "data.root", "data.vocab_path", "data.split_manifest", "data.subset_manifest",
    "training.epochs", "training.batch_size", "training.lr",
)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def value_at(config: dict, dotted: str):
    value = config
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate_config(config: dict) -> list[str]:
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    required = {"experiment", "variant", "seed", "model", "data", "training", "distillation", "output"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
    if config["experiment"] not in {f"E{i}" for i in range(1, 8)}:
        raise ValueError("experiment must be E1 through E7")
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for section in ("model", "data", "training", "distillation", "output"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
    if config["data"].get("image_size") != [32, 128]:
        raise ValueError("This protocol requires image_size: [32, 128]")
    for field in ("training.epochs", "training.batch_size"):
        value = value_at(config, field)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{field} must be a positive integer or null")
    lr = value_at(config, "training.lr")
    if lr is not None and (isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0):
        raise ValueError("training.lr must be positive or null")
    kd = config["distillation"]
    if type(kd.get("enabled")) is not bool:
        raise ValueError("distillation.enabled must be a boolean")
    if kd["enabled"]:
        for key in ("temperature", "lambda"):
            value = kd.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"distillation.{key} must be positive")
    if not isinstance(config["output"].get("root"), str) or not config["output"]["root"].strip():
        raise ValueError("output.root must be a nonempty path")
    unresolved = [field for field in UNSET_FIELDS if value_at(config, field) is None]
    if kd["enabled"] and not config["model"].get("teacher_checkpoint"):
        unresolved.append("model.teacher_checkpoint")
    return unresolved
