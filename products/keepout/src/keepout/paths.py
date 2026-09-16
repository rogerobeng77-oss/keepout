"""Where the bundled data lives, resolved the same way everywhere.

This exists because of a real build failure. `cli.py` computed its sample
directory as `Path(__file__).parents[2] / "data" / "samples"`, which is correct
when the package is imported from the source tree and nonsense once pip has
installed it into `site-packages`. The Docker build then failed at the step that
bakes the demo, with "no sample cell-alert", while every local run was fine.

So paths are resolved once, here, by trying in order:

1. the environment variable, which is what the container sets;
2. the source-tree location, for a development checkout or an editable install;
3. the installed-package location, for a plain `pip install`.

A missing directory is returned rather than raising, because the caller usually
wants to create it.
"""

from __future__ import annotations

import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent

# src/keepout/paths.py -> src/keepout -> src -> <product root>
_SOURCE_ROOT = _MODULE_DIR.parents[1]
# site-packages/keepout/paths.py -> site-packages/keepout
_INSTALLED_ROOT = _MODULE_DIR


def _resolve(env: str, *relative: str) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override)
    candidates = [
        _SOURCE_ROOT.joinpath(*relative),
        _INSTALLED_ROOT.joinpath(*relative),
        Path("/app/products/keepout").joinpath(*relative),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def product_root() -> Path:
    return _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else _INSTALLED_ROOT


def samples_dir() -> Path:
    return _resolve("KEEPOUT_SAMPLES_DIR", "data", "samples")


def demo_dir() -> Path:
    return _resolve("KEEPOUT_DEMO_DIR", "data", "demo")


def source_dir() -> Path:
    return _resolve("KEEPOUT_SOURCE_DIR", "data", "source")


def static_dir() -> Path:
    return _resolve("KEEPOUT_STATIC_DIR", "static")


def eval_dir() -> Path:
    return _resolve("KEEPOUT_EVAL_DIR", "eval", "results")


def describe() -> dict[str, str]:
    """For `/version` and for working out why a container cannot find its clips."""
    return {
        "samples": str(samples_dir()),
        "demo": str(demo_dir()),
        "static": str(static_dir()),
        "source": str(source_dir()),
    }
