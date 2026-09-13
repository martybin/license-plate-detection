"""Report exactly why prepare_dataset skipped an image, and how often.

prepare_dataset prints a single `Failed: N` count per split, which is enough to
know the dataset is healthy but not enough to tell a missing XML apart from an
annotation whose character names normalise to nothing. This walks the same code
paths and breaks the number down.

    python -m tools.diagnose_failures /path/to/Bistun-kavir/train
    python -m tools.diagnose_failures /path/to/train /path/to/validation
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from training.prepare_dataset import (
    IMG_EXTS,
    _is_plate_object,
    clean_plate_text,
    find_xml,
    parse_xml,
    usable_region,
)
from utils.plate_utils import is_valid_iran_plate

# How many concrete examples to print per failure reason. Enough to spot the
# pattern by eye without burying the summary.
EXAMPLES = 5


def classify(img_path: Path) -> tuple[str, str]:
    """Return (reason, detail) for one image, mirroring rename_images_using_xml."""
    xml_path = find_xml(img_path)
    if xml_path is None:
        return "no_xml", ""

    try:
        ET.parse(xml_path)
    except (ET.ParseError, OSError) as exc:
        return "xml_unparseable", type(exc).__name__

    boxes, _ = parse_xml(xml_path)
    if not boxes:
        return "xml_has_no_objects", ""

    chars = [b for b in boxes if not _is_plate_object(b.name)]
    if not chars:
        return "only_whole_plate_box", f"{len(boxes)} object(s)"

    raw = "".join(b.name for b in chars)
    trusted, _cropped = usable_region(chars)
    plate = clean_plate_text("".join(b.name for b in trusted))

    if len(plate) < 5:
        # The interesting case: objects existed but normalisation emptied them.
        dropped = [b.name for b in chars if not clean_plate_text(b.name)]
        if dropped:
            return "names_normalise_to_nothing", f"raw={raw!r} -> {plate!r} dropped={dropped[:6]}"
        return "plate_too_short", f"raw={raw!r} -> {plate!r} ({len(plate)} chars)"

    if not is_valid_iran_plate(plate):
        # Not a failure for prepare_dataset, but worth knowing: it becomes a
        # training label the recognizer can never match against a real plate.
        return "accepted_but_malformed", f"{plate!r}"
    return "ok", plate


def main() -> None:
    # The report is full of Persian glyph names; a cp1252 console would die on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Break down prepare_dataset OCR failures")
    parser.add_argument("folders", type=Path, nargs="+", help="plate-crop folders (train/validation/test)")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)

    for folder in args.folders:
        if not folder.exists():
            raise SystemExit(f"Folder not found: {folder}")
        print(f"Scanning {folder} ...")
        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            reason, detail = classify(img_path)
            counts[reason] += 1
            if len(examples[reason]) < EXAMPLES:
                examples[reason].append(f"{img_path.name}  {detail}".rstrip())

    total = sum(counts.values())
    ok = counts["ok"] + counts["accepted_but_malformed"]
    print(f"\n{'=' * 72}")
    print(f"Scanned {total} image(s) | kept {ok} | skipped {total - ok}")
    print(f"{'=' * 72}\n")

    for reason, count in counts.most_common():
        share = count / total if total else 0.0
        tag = "" if reason == "ok" else "  <-- skipped by prepare_dataset"
        if reason == "accepted_but_malformed":
            tag = "  <-- kept, but not a valid Iranian plate layout"
        print(f"{reason:<30} {count:>7}  {share:>6.1%}{tag}")
        if reason != "ok":
            for example in examples[reason]:
                print(f"    {example}")
        print()


if __name__ == "__main__":
    main()
