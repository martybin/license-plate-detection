from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class VehicleDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    plate TEXT PRIMARY KEY,
                    driver_name TEXT,
                    national_id TEXT,
                    truck_id TEXT,
                    vehicle_model TEXT,
                    company TEXT,
                    allowed INTEGER DEFAULT 1,
                    note TEXT
                )
                """
            )
            conn.commit()

    def lookup(self, plate: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT plate, driver_name, national_id, truck_id,
                       vehicle_model, company, allowed, note
                FROM vehicles WHERE plate = ?
                """,
                (plate,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def upsert(
        self,
        plate: str,
        driver_name: str = "",
        national_id: str = "",
        truck_id: str = "",
        vehicle_model: str = "",
        company: str = "",
        allowed: int = 1,
        note: str = "",
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO vehicles
                    (plate, driver_name, national_id, truck_id,
                     vehicle_model, company, allowed, note)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plate) DO UPDATE SET
                    driver_name   = excluded.driver_name,
                    national_id   = excluded.national_id,
                    truck_id      = excluded.truck_id,
                    vehicle_model = excluded.vehicle_model,
                    company       = excluded.company,
                    allowed       = excluded.allowed,
                    note          = excluded.note
                """,
                (
                    plate,
                    driver_name,
                    national_id,
                    truck_id,
                    vehicle_model,
                    company,
                    allowed,
                    note,
                ),
            )
            conn.commit()

    def seed_demo(self) -> None:
        demos = [
            ("12ب34567", "علی محمدی", "0012345678", "TRK-001", "بنز ۱۹۲۳", "معدن سنگان", 1, ""),
            ("34ج67890", "رضا کریمی", "0098765432", "TRK-014", "ولوو FH", "معدن سنگان", 1, ""),
            ("56د12345", "حسین رضایی", "1122334455", "TRK-022", "مان TGA", "معدن گل‌گهر", 0, "غیرمجاز"),
        ]
        for item in demos:
            self.upsert(*item)
