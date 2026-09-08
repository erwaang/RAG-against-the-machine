"""Entry point for ``python -m src <command>``."""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "hf-cache"))

import fire  # noqa: E402
from src.cli import CLI  # noqa: E402


if __name__ == "__main__":
    fire.Fire(CLI)
