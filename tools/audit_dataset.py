"""Report OCR label failures without torch or changes to the dataset."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from utils.plate_utils import PLATE_LETTERS, label_from_filename, is_valid_iran_plate

def audit(root):
    counts, letters = Counter(), Counter()
    rejected, labels = [], {}
    charset = set("0123456789" + PLATE_LETTERS)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = label_from_filename(path)
        usable = 5 <= len(label) <= 10 and set(label) <= charset
        counts["images"] += 1
        counts["usable"] += int(usable)
        counts["valid_full_plate"] += int(is_valid_iran_plate(label))
        if not usable and len(rejected) < 100:
            rejected.append({"file": str(path.relative_to(root)), "label": label})
        if usable:
            letters.update(c for c in label if not c.isdigit())
        split = path.relative_to(root).parts[0] if path.parent != root else "flat"
        labels.setdefault(split, set()).add(label)
    overlaps = {a + "/" + b: len(labels[a] & labels[b])
                for a in ("train", "val", "test") for b in ("train", "val", "test")
                if a < b and a in labels and b in labels}
    return {"counts": dict(counts),
            "usable_ratio": counts["usable"] / counts["images"] if counts["images"] else None,
            "letters": dict(letters), "overlapping_labels": overlaps,
            "rejected_examples": rejected,
            "note": "Label audit only; this does not measure accuracy or prove overfitting."}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/ocr_dataset"))
    p.add_argument("--output", type=Path, default=Path("reports/dataset-audit.json"))
    args = p.parse_args()
    if not args.data.is_dir():
        p.error(f"Missing dataset: {args.data}")
    report = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")

if __name__ == "__main__":
    main()
