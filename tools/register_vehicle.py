"""Manage the driver/vehicle registry that the gate monitor displays.

Without this the database only ever holds the three demo rows, so a real gate
has nothing to show. Run it from the project root:

    python -m tools.register_vehicle add --plate 12ب34567 --driver "علی محمدی" ...
    python -m tools.register_vehicle import drivers.csv
    python -m tools.register_vehicle list
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

from utils.database import VehicleDB
from utils.plate_utils import format_plate_display, is_valid_iran_plate, normalize_iran_plate

CSV_FIELDS = ["plate", "driver_name", "national_id", "truck_id", "vehicle_model", "company", "allowed", "note"]


def open_db(config_path: str) -> VehicleDB:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return VehicleDB(cfg["database"]["path"])


def _clean_plate(raw: str, strict: bool = True) -> str:
    plate = normalize_iran_plate(raw)
    if not is_valid_iran_plate(plate):
        message = f"'{raw}' is not a valid Iranian plate (expected e.g. 12ب34567)"
        if strict:
            raise SystemExit(f"error: {message}")
        print(f"  skipped: {message}", file=sys.stderr)
        return ""
    return plate


def cmd_add(args) -> None:
    plate = _clean_plate(args.plate)
    with open_db(args.config) as db:
        db.upsert(
            plate=plate,
            driver_name=args.driver,
            national_id=args.national_id,
            truck_id=args.truck_id,
            vehicle_model=args.model,
            company=args.company,
            allowed=0 if args.denied else 1,
            note=args.note,
        )
    print(f"saved {format_plate_display(plate)}  ({'DENIED' if args.denied else 'ALLOWED'})")


def cmd_remove(args) -> None:
    plate = _clean_plate(args.plate)
    with open_db(args.config) as db:
        if db.lookup(plate) is None:
            raise SystemExit(f"error: {plate} is not registered")
        db.delete(plate)
    print(f"removed {format_plate_display(plate)}")


def cmd_list(args) -> None:
    with open_db(args.config) as db:
        rows = db.all()
    if not rows:
        print("registry is empty")
        return
    for row in rows:
        status = "ALLOWED" if row["allowed"] else "DENIED "
        print(
            f"{format_plate_display(row['plate']):>22}  {status}  "
            f"{row['driver_name'] or '-':<20} {row['truck_id'] or '-':<10} {row['company'] or '-'}"
        )
    print(f"\n{len(rows)} vehicle(s)")


def cmd_import(args) -> None:
    path = Path(args.csv_path)
    if not path.exists():
        raise SystemExit(f"error: {path} not found")

    added = skipped = 0
    with open_db(args.config) as db, path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"plate"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"error: CSV must have a 'plate' column; found {reader.fieldnames}")

        for row in reader:
            plate = _clean_plate(row.get("plate", ""), strict=False)
            if not plate:
                skipped += 1
                continue
            allowed = str(row.get("allowed", "1")).strip().lower()
            db.upsert(
                plate=plate,
                driver_name=row.get("driver_name", ""),
                national_id=row.get("national_id", ""),
                truck_id=row.get("truck_id", ""),
                vehicle_model=row.get("vehicle_model", ""),
                company=row.get("company", ""),
                allowed=0 if allowed in {"0", "false", "no", "غیرمجاز"} else 1,
                note=row.get("note", ""),
            )
            added += 1
    print(f"imported {added} vehicle(s), skipped {skipped}")


def cmd_export(args) -> None:
    with open_db(args.config) as db:
        rows = db.all()
    with Path(args.csv_path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} vehicle(s) to {args.csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the gate's vehicle registry")
    parser.add_argument("--config", default="configs/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="add or update one vehicle")
    add.add_argument("--plate", required=True)
    add.add_argument("--driver", default="")
    add.add_argument("--national-id", default="")
    add.add_argument("--truck-id", default="")
    add.add_argument("--model", default="")
    add.add_argument("--company", default="")
    add.add_argument("--note", default="")
    add.add_argument("--denied", action="store_true", help="mark the vehicle as not allowed")
    add.set_defaults(func=cmd_add)

    remove = sub.add_parser("remove", help="delete one vehicle")
    remove.add_argument("--plate", required=True)
    remove.set_defaults(func=cmd_remove)

    listing = sub.add_parser("list", help="show every registered vehicle")
    listing.set_defaults(func=cmd_list)

    importer = sub.add_parser("import", help="bulk import from a CSV file")
    importer.add_argument("csv_path")
    importer.set_defaults(func=cmd_import)

    exporter = sub.add_parser("export", help="write the registry to a CSV file")
    exporter.add_argument("csv_path")
    exporter.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
