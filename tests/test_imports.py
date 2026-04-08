"""Smoke tests for package imports and optional extras."""

import builtins
import importlib

import pytest


def test_package_import_smoke():
    package = importlib.import_module("simplex_codes")
    assert package.__version__ == "0.1.0"


def test_visualization_module_import_smoke():
    module = importlib.import_module("simplex_codes.visualization")
    assert hasattr(module, "plot_trajectory")


def test_visualization_requires_optional_dependency(monkeypatch):
    module = importlib.import_module("simplex_codes.visualization")

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot" or name.startswith("matplotlib"):
            raise ImportError("mocked missing matplotlib")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="\\[viz\\]"):
        module.plot_trajectory([[0, 0], [1, 1]])
