from __future__ import annotations

import argparse
from pathlib import Path

from inference.realtime import RealtimeLPR


def main() -> None:
    parser = argparse.ArgumentParser(description="Iranian License Plate Recognition - Mine Gate")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    app = RealtimeLPR(config_path)
    app.run()


if __name__ == "__main__":
    main()
