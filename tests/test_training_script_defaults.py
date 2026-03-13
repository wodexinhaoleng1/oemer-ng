"""Tests for training/evaluation script dataset defaults."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _get_add_argument_defaults(script_path: Path) -> dict[str, object]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant):
            continue
        if not isinstance(node.args[0].value, str):
            continue

        arg_name = node.args[0].value
        for kw in node.keywords:
            if kw.arg == "default":
                defaults[arg_name] = ast.literal_eval(kw.value)

    return defaults


def test_train_model_defaults_to_ds2_dataset():
    defaults = _get_add_argument_defaults(REPO_ROOT / "examples" / "train_model.py")
    assert defaults["--dataset_path"] == "data/sample_ds2"
    assert defaults["--dataset_type"] == "ds2"


def test_evaluate_model_defaults_to_ds2_dataset_type():
    defaults = _get_add_argument_defaults(REPO_ROOT / "scripts" / "evaluate_model.py")
    assert defaults["--dataset_type"] == "ds2"
