from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional

_COLUMNS = (
    "plate, driver_name, national_id, truck_id, vehicle_model, company, allowed, note"
)


class VehicleDB:
    """Driver/vehicle registry for the gate display.

    Holds one long-lived connection instead of opening a new one per lookup. The
    previous version opened a connection on every frame and never closed it --
    `with sqlite3.connect(...)` commits on exit but does not close -- which leaked
    a file descriptor per frame over a shift.
    """

    def __init__(self, db_path: str | Path, cache_size: int = 256) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Optional[dict]] = {}
        self._cache_size = cache_size

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps reads from blocking while an operator edits the registry.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
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

    def lookup(self, plate: str) -> Optional[dict]:
        if not plate:
            return None
        if plate in self._cache:
            return self._cache[plate]

        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM vehicles WHERE plate = ?", (plate,)
            ).fetchone()

        result = dict(row) if row else None
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[plate] = result
        return result

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
        with self._lock, self._conn:
            self._conn.execute(
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
                (plate, driver_name, national_id, truck_id, vehicle_model, company, allowed, note),
            )
        self._cache.pop(plate, None)

    def delete(self, plate: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM vehicles WHERE plate = ?", (plate,))
        self._cache.pop(plate, None)

    def all(self) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM vehicles ORDER BY plate"
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0])

    def seed_demo(self) -> None:
        """Insert sample rows, never overwriting a plate already on file.

        This used to run an upsert on every startup, which silently reset three
        real driver records each time the gate was restarted.
        """
        demos = [
            ("12ب34567", "علی محمدی", "0012345678", "TRK-001", "بنز ۱۹۲۳", "معدن سنگان", 1, ""),
            ("34ج67890", "رضا کریمی", "0098765432", "TRK-014", "ولوو FH", "معدن سنگان", 1, ""),
            ("56د12345", "حسین رضایی", "1122334455", "TRK-022", "مان TGA", "معدن گل‌گهر", 0, "غیرمجاز"),
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                f"INSERT OR IGNORE INTO vehicles ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                demos,
            )
        self._cache.clear()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "VehicleDB":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
