import os
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, replace
from typing import Any, Literal

import torch
import yaml

TaskType = Literal["reg", "cls", "multi"]


@dataclass(frozen=True)
class Opt:
    # --- General ---
    name: str
    neptune_project_name: str
    neptune_group_tags: list[str]

    # --- Data / paths ---
    dataroot: str
    reg_feature: str
    cls_feature: str
    features: list[str]
    masked_features: list[str]
    checkpoints_path: str
    window: int
    window_stride: int | None = None
    drop_tail: bool = True
    split_ratio: float = 0.9
    batch_size: int = 8
    num_workers: int = 8
    return_window: bool = False
    shuffle: bool = False

    # --- Model / tokenizer ---
    in_channels: int | None = None
    d_model: int = 128
    dow_embedding_dim: int = 8
    k: int = 5  # kernel size
    s: int | None = None  # stride; if None -> k (set in __post_init__)
    nhead: int = 4
    t_num_layers: int = 2
    head_hidden: int = 0  # 0 = single linear
    lambda_reg: float = 1.0  # Used in multitask schema
    lambda_cls: float = 1.0  # Used in multitask schema

    # --- Task / training ---
    epochs: int = 50
    phase: Literal["train", "test"] = "train"
    task_type: TaskType = "reg"
    lr: float = 3e-4
    margin_eps: int | None = None
    log_step: int = 10
    save_step: int = 5

    # --- testing ---
    load_path: str = ""

    # --- System ---
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Derived / validation ---
    def __post_init__(self):
        # Because dataclass is frozen, use object.__setattr__
        if self.s is None:
            object.__setattr__(self, "s", self.k)
        if self.window_stride is None:
            object.__setattr__(self, "window_stride", self.window)
        if self.in_channels is None:
            if "dow" in self.features:
                in_channels = len(self.features) + self.dow_embedding_dim - 1
            else:
                in_channels = len(self.features)
            object.__setattr__(self, "in_channels", in_channels)

        # Normalize device to torch.device
        dev = (
            torch.device(self.device)
            if not isinstance(self.device, torch.device)
            else self.device
        )
        object.__setattr__(self, "device", dev)

        checkpoints_exist_ok = self.phase == "test"
        os.makedirs(
            os.path.join(self.checkpoints_path, self.name),
            exist_ok=checkpoints_exist_ok,
        )

        # Basic validation
        if not os.path.exists(self.dataroot):
            raise ValueError(f"path {self.dataroot} does not exist")
        if self.split_ratio > 1 or self.split_ratio < 0:
            raise ValueError(
                f"split_ratio must be between 0 and 1, got {self.split_ratio}"
            )
        n_channels = (
            len(self.features)
            if "dow" not in self.features
            else len(self.features) + self.dow_embedding_dim - 1
        )
        if self.in_channels != n_channels:
            raise ValueError(
                f"in_channels should be the same as len(features), got in_channels: {self.in_channels}, features: {self.features}"
            )
        if self.window < 0:
            raise ValueError(f"Window size can't be lower than 0, got {self.window}")
        if self.window_stride is None or self.window_stride < 0:
            raise ValueError(
                f"Stride size can't be lower than 0, got {self.window_stride}"
            )
        if self.task_type not in ("reg", "cls", "multi"):
            raise ValueError(
                f"task_type must be one of ['reg','cls','multi'], got {self.task_type}"
            )
        if self.k <= 0 or self.s <= 0:
            raise ValueError("k and s must be positive integers")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("nhead must divide d_model")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")

        print(self.summary())

    # --- Convenience helpers ---
    def to_dict(self) -> dict:
        d = asdict(self)
        d["device"] = str(self.device)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Opt":
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "Opt":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"Configuration at {path} must be a mapping.")
        data = dict(data)
        data = cls._expand_test_config(data, path)
        return cls.from_dict(data)

    @classmethod
    def _required_field_names(cls) -> set[str]:
        return {
            name
            for name, field in cls.__dataclass_fields__.items()
            if field.default is MISSING and field.default_factory is MISSING
        }

    @classmethod
    def _expand_test_config(cls, data: dict, source_path: str) -> dict:
        if data.get("phase") != "test":
            return data

        required_fields = cls._required_field_names()
        missing_required = required_fields - data.keys()

        if not missing_required:
            if "load_path" not in data:
                checkpoints_root = data.get("checkpoints_path", "checkpoints")
                data["load_path"] = os.path.join(
                    checkpoints_root, data["name"], "checkpoint_latest.pt"
                )
            return data

        if "name" not in data:
            raise ValueError(
                f"Test configuration {source_path} must include the 'name' of the run."
            )

        config_candidates = []

        load_path = data.get("load_path")
        if load_path:
            config_candidates.append(
                os.path.join(os.path.dirname(load_path), "config.yaml")
            )

        checkpoints_root = data.get("checkpoints_path", "checkpoints")
        config_candidates.append(
            os.path.join(checkpoints_root, data["name"], "config.yaml")
        )

        training_config_path = next(
            (candidate for candidate in config_candidates if os.path.exists(candidate)),
            None,
        )
        if training_config_path is None:
            raise FileNotFoundError(
                f"Unable to locate stored training config for '{data['name']}'. "
                f"Tried: {', '.join(config_candidates)}"
            )

        with open(training_config_path, "r") as f:
            base_data = yaml.safe_load(f) or {}

        if base_data.get("name") and base_data["name"] != data["name"]:
            raise ValueError(
                f"Stored training config at {training_config_path} belongs to run "
                f"'{base_data['name']}', but test config requested '{data['name']}'."
            )

        merged = {**base_data, **data}
        merged.setdefault(
            "checkpoints_path", base_data.get("checkpoints_path", checkpoints_root)
        )
        merged.setdefault(
            "load_path",
            os.path.join(
                merged["checkpoints_path"], merged["name"], "checkpoint_latest.pt"
            ),
        )

        remaining_missing = required_fields - merged.keys()
        if remaining_missing:
            raise ValueError(
                f"Test configuration {source_path} is missing required settings even "
                f"after merging with {training_config_path}: {sorted(remaining_missing)}"
            )

        return merged

    def updated(self, **overrides) -> "Opt":
        """Return a new Opt with overrides (immutability-friendly)."""
        return replace(self, **overrides)

    def summary(self) -> str:
        # Gather all settings in sections
        settings = self.to_dict()
        lines = []
        lines.append("=" * 50)
        lines.append(f"{'FiT Configuration Summary':^50}")
        lines.append("=" * 50)
        # General
        lines.append(f"{'General':<15}: name = {settings['name']}")
        # Data / Paths
        lines.append(f"{'Data root':<15}: {settings['dataroot']}")
        lines.append(f"{'Features':<15}: {settings['features']}")
        lines.append(f"{'Masked feats':<15}: {settings['masked_features']}")
        lines.append(f"{'Checkpoints':<15}: {settings['checkpoints_path']}")
        lines.append(
            f"{'Window':<15}: {settings['window']} (stride={settings['window_stride']}, drop_tail={settings['drop_tail']})"
        )
        lines.append(f"{'Split ratio':<15}: {settings['split_ratio']}")
        # Model / Tokenizer
        lines.append("-" * 50)
        lines.append(
            f"{'Model':<15}: in_channels={settings['in_channels']}, d_model={settings['d_model']}, k={settings['k']}, s={settings['s']}"
        )
        lines.append(
            f"{'Embedding':<15}: dow_embedding_dim={settings['dow_embedding_dim']}"
        )
        lines.append(
            f"{'Heads/Layers':<15}: nhead={settings['nhead']}, t_num_layers={settings['t_num_layers']}, head_hidden={settings['head_hidden']}"
        )
        # Task / Training
        lines.append("-" * 50)
        lines.append(
            f"{'Task':<15}: phase={settings['phase']}, type={settings['task_type']}, lr={settings['lr']}, margin_eps={settings['margin_eps']}"
        )
        # System
        lines.append(f"{'Device':<15}: {settings['device']}")
        lines.append("=" * 50)
        return "\n".join(lines)
