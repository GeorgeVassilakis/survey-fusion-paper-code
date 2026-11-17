from __future__ import annotations

import ast
from pathlib import Path


UNCERTAINTY_NAMES = {"sig", "sigma", "uncertainty", "ivar"}
CLAMP_NAMES = {"clamp", "clamp_min", "clamp_max", "clip"}


def _is_uncertainty_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and any(
        token in node.id.lower() for token in UNCERTAINTY_NAMES
    )


def test_source_and_diagnostic_code_do_not_clamp_uncertainties() -> None:
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for directory in (root / "src", root / "scripts"):
        for path in directory.rglob("*.py"):
            if path.name.startswith("._"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr in CLAMP_NAMES
                    and _is_uncertainty_name(function.value)
                ):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr in CLAMP_NAMES
                    and node.args
                    and _is_uncertainty_name(node.args[0])
                ):
                    violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, f"Uncertainty clamps found: {violations}"
