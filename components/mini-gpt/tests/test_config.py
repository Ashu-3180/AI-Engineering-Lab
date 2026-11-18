from pathlib import Path

import pytest
import torch

from mini_gpt.batching import make_batch_from_config
from mini_gpt.config import load_data_config, resolve_component_path
from mini_gpt.dataset import split_from_config

VALID = """[data]
dataset_id = "tinyshakespeare"
source_url = "https://example.test/input.txt"
raw_data_dir = "data/raw"
processed_data_dir = "data/processed"
provenance_dir = "data/provenance"
train_ratio = 0.8
default_context_length = 2
default_batch_size = 3
seed = 9
"""


def write_config(tmp_path, body):
    path = tmp_path / "data.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_toml_configuration_controls_split_and_batch(tmp_path):
    config = load_data_config(write_config(tmp_path, VALID))
    split = split_from_config(list(range(10)), config)
    assert split.train_boundary == 8
    first = make_batch_from_config(split.train, config=config)
    second = make_batch_from_config(split.train, config=config)
    assert first[0].shape == first[1].shape == (3, 2)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))


@pytest.mark.parametrize(
    "body",
    [
        "[data\n",
        VALID.replace("default_batch_size = 3\n", ""),
        VALID.replace("train_ratio = 0.8", "train_ratio = 1.0"),
        VALID.replace("default_context_length = 2", "default_context_length = 0"),
        VALID.replace("default_batch_size = 3", "default_batch_size = 0"),
    ],
)
def test_invalid_or_missing_toml_settings_are_rejected(tmp_path, body):
    with pytest.raises(ValueError):
        load_data_config(write_config(tmp_path, body))


def test_component_relative_paths_are_safe(tmp_path):
    root = tmp_path / "component"
    root.mkdir()
    assert resolve_component_path(root, Path("data/raw")) == root / "data" / "raw"
    assert resolve_component_path(root, Path("nested/data")) == root / "nested" / "data"
    with pytest.raises(ValueError):
        resolve_component_path(root, Path("../outside"))
    with pytest.raises(ValueError):
        resolve_component_path(root, Path("C:/outside"))


def test_windows_unc_path_is_not_treated_as_safe_relative_path(tmp_path):
    root = tmp_path / "component"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_component_path(root, Path("\\\\server\\share"))
