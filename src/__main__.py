"""Entry point for ``python -m src <command>``."""

import fire
from src.cli import CLI


if __name__ == "__main__":
    fire.Fire(CLI)
