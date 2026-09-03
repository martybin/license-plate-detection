"""Vehicle registry behaviour."""
from __future__ import annotations

import sqlite3

import pytest

from utils.database import VehicleDB


class TestBasics:
    def test_creates_nested_path(self, tmp_path):
        db = VehicleDB(tmp_path / "a" / "b" / "vehicles.db")
        assert (tmp_path / "a" / "b" / "vehicles.db").exists()
        db.close()

    def test_upsert_then_lookup(self, db):
        db.upsert("12ب34567", driver_name="علی محمدی", truck_id="TRK-001", company="معدن سنگان")
        row = db.lookup("12ب34567")
        assert row["driver_name"] == "علی محمدی"
        assert row["truck_id"] == "TRK-001"
        assert row["allowed"] == 1

    def test_lookup_miss_and_empty(self, db):
        assert db.lookup("99ق99999") is None
        assert db.lookup("") is None

    def test_upsert_updates_in_place(self, db):
        db.upsert("12ب34567", driver_name="اول")
        db.upsert("12ب34567", driver_name="دوم")
        assert db.count() == 1
        assert db.lookup("12ب34567")["driver_name"] == "دوم"

    def test_denied_vehicle(self, db):
        db.upsert("56د12345", driver_name="x", allowed=0, note="مدارک ناقص")
        row = db.lookup("56د12345")
        assert row["allowed"] == 0
        assert row["note"] == "مدارک ناقص"

    def test_delete(self, db):
        db.upsert("12ب34567", driver_name="x")
        db.delete("12ب34567")
        assert db.lookup("12ب34567") is None
        assert db.count() == 0

    def test_all_is_sorted(self, db):
        for plate in ("34ج67890", "12ب34567", "56د12345"):
            db.upsert(plate)
        assert [r["plate"] for r in db.all()] == sorted(["34ج67890", "12ب34567", "56د12345"])


class TestCache:
    def test_upsert_invalidates_cache(self, db):
        db.upsert("12ب34567", driver_name="اول")
        assert db.lookup("12ب34567")["driver_name"] == "اول"  # populates cache
        db.upsert("12ب34567", driver_name="دوم")
        assert db.lookup("12ب34567")["driver_name"] == "دوم"

    def test_delete_invalidates_cache(self, db):
        db.upsert("12ب34567")
        db.lookup("12ب34567")
        db.delete("12ب34567")
        assert db.lookup("12ب34567") is None

    def test_cache_is_bounded(self, tmp_path):
        db = VehicleDB(tmp_path / "v.db", cache_size=4)
        for i in range(50):
            db.lookup(f"1{i % 9}ب3456{i % 9}")
        assert len(db._cache) <= 4
        db.close()

    def test_misses_are_cached_too(self, db):
        assert db.lookup("99ق99999") is None
        assert "99ق99999" in db._cache


class TestSeedDemo:
    def test_seeds_when_empty(self, db):
        db.seed_demo()
        assert db.count() == 3

    def test_never_overwrites_a_real_record(self, db):
        """Seeding ran on every boot and used to reset three real driver rows."""
        db.seed_demo()
        db.upsert("12ب34567", driver_name="نام واقعی", company="شرکت واقعی", allowed=0)

        db.seed_demo()

        row = db.lookup("12ب34567")
        assert row["driver_name"] == "نام واقعی"
        assert row["company"] == "شرکت واقعی"
        assert row["allowed"] == 0
        assert db.count() == 3

    def test_is_idempotent(self, db):
        for _ in range(3):
            db.seed_demo()
        assert db.count() == 3


class TestLifecycle:
    def test_data_survives_reopen(self, tmp_path):
        path = tmp_path / "v.db"
        db = VehicleDB(path)
        db.upsert("12ب34567", driver_name="علی")
        db.close()

        reopened = VehicleDB(path)
        assert reopened.lookup("12ب34567")["driver_name"] == "علی"
        reopened.close()

    def test_context_manager_closes(self, tmp_path):
        with VehicleDB(tmp_path / "v.db") as db:
            db.upsert("12ب34567")
        with pytest.raises(sqlite3.ProgrammingError):
            db.count()

    def test_single_connection_is_reused(self, db):
        """One long-lived connection; the old code opened one per frame and leaked it."""
        before = db._conn
        for _ in range(100):
            db.lookup("12ب34567")
        assert db._conn is before
