import platform
import sys

import torch


def test_frozen_environment_versions_and_gil() -> None:
    assert sys.version_info[:3] == (3, 15, 2)
    assert platform.python_implementation() == "CPython"
    assert sys._is_gil_enabled()
    assert torch.__version__.split("+")[0] == "2.15.0"


def test_cpu_tensor_and_autograd_are_usable() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = (values * 2).sum()
    result.backward()
    assert result.device.type == "cpu"
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.equal(values.grad, torch.tensor([2.0, 2.0, 2.0]))
