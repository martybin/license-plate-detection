from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    # Allows `python /path/to/main.py` from any working directory.
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.realtime import RealtimeLPR  # noqa: E402


def check_weights(config_path: Path) -> None:
    """Fail with an actionable message instead of a torch stack trace."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    missing = []
    for name in ("detector", "recognizer"):
        path = Path(cfg[name]["model_path"])
        if not path.exists():
            missing.append((name, path))

    if missing:
        details = "\n".join(f"  - {name}: {path}" for name, path in missing)
        # A bare traceback on a gate console tells the operator nothing useful.
        raise SystemExit(
            f"Missing model weights:\n{details}\n\n"
            "Train them first:\n"
            "  python -m training.prepare_dataset --root /mnt/g/Bistun-kavir\n"
            "  python -m training.train_detector\n"
            "  python -m training.train_recognizer"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Iranian License Plate Recognition - Mine Gate")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--source", type=str, default=None, help="Override camera source (index, file or rtsp URL)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    check_weights(config_path)

    app = RealtimeLPR(config_path)
    if args.source is not None:
        # Digits mean a local camera index; anything else is a path or URL.
        app.grabber.source = int(args.source) if args.source.isdigit() else args.source
    app.run()


if __name__ == "__main__":
    main()
