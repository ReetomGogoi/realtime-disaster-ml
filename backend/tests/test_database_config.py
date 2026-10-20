"""Database configuration tests; no database connections are opened."""

import tempfile
import unittest
from pathlib import Path

from app.database.config import resolve_database_url


class DatabaseConfigTests(unittest.TestCase):
    def test_local_default_uses_existing_sqlite_location(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "backend" / "data"

            url = resolve_database_url({}, data_dir)

            self.assertEqual(url, f"sqlite:///{data_dir / 'disasters.db'}")
            self.assertTrue(data_dir.is_dir())

    def test_local_default_does_not_overwrite_existing_sqlite_database(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "backend" / "data"
            data_dir.mkdir(parents=True)
            database = data_dir / "disasters.db"
            original = b"existing SQLite database content"
            database.write_bytes(original)

            url = resolve_database_url({}, data_dir)

            self.assertEqual(url, f"sqlite:///{database}")
            self.assertEqual(database.read_bytes(), original)

    def test_explicit_local_sqlite_url_is_preserved_without_creating_default_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "unused"
            configured = "sqlite:////tmp/realtime-disaster-local.db"

            url = resolve_database_url({"DATABASE_URL": configured}, data_dir)

            self.assertEqual(url, configured)
            self.assertFalse(data_dir.exists())

    def test_postgresql_url_never_creates_local_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "unused"
            configured = "postgresql://user:password@example.invalid/disasters"

            url = resolve_database_url({"DATABASE_URL": configured}, data_dir)

            self.assertEqual(url, configured)
            self.assertFalse(data_dir.exists())

    def test_local_neon_psycopg_url_is_normalized_without_creating_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "unused"
            configured = (
                "postgresql+psycopg://owner:p%40ss@ep-test.neon.tech/neondb"
                "?sslmode=require&channel_binding=require"
            )

            url = resolve_database_url({"DATABASE_URL": configured}, data_dir)

            self.assertEqual(
                url,
                "postgresql+psycopg2://owner:p%40ss@ep-test.neon.tech/neondb"
                "?sslmode=require&channel_binding=require",
            )
            self.assertFalse(data_dir.exists())

    def test_vercel_requires_database_url(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"

            with self.assertRaisesRegex(RuntimeError, "persistent PostgreSQL"):
                resolve_database_url({"VERCEL": "1"}, data_dir)

            self.assertFalse(data_dir.exists())

    def test_vercel_rejects_empty_database_url(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"

            with self.assertRaisesRegex(RuntimeError, "persistent PostgreSQL"):
                resolve_database_url({"VERCEL": "1", "DATABASE_URL": "  "}, data_dir)

            self.assertFalse(data_dir.exists())

    def test_vercel_rejects_sqlite_without_creating_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"

            with self.assertRaisesRegex(RuntimeError, "require a PostgreSQL DATABASE_URL"):
                resolve_database_url(
                    {"VERCEL": "1", "DATABASE_URL": "sqlite:////tmp/disasters.db"},
                    data_dir,
                )

            self.assertFalse(data_dir.exists())

    def test_vercel_accepts_standard_postgresql_url_without_creating_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"
            configured = "postgresql://user:password@example.invalid/disasters"

            url = resolve_database_url(
                {"VERCEL": "1", "DATABASE_URL": configured}, data_dir
            )

            self.assertEqual(url, configured)
            self.assertFalse(data_dir.exists())

    def test_vercel_neon_psycopg_url_uses_installed_psycopg2_driver(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"
            configured = (
                "postgresql+psycopg://owner:secret@ep-test.neon.tech/neondb"
                "?sslmode=require"
            )

            url = resolve_database_url(
                {"VERCEL": "1", "DATABASE_URL": configured}, data_dir
            )

            self.assertEqual(
                url,
                "postgresql+psycopg2://owner:secret@ep-test.neon.tech/neondb"
                "?sslmode=require",
            )
            self.assertFalse(data_dir.exists())

    def test_vercel_accepts_explicit_psycopg2_url(self):
        with tempfile.TemporaryDirectory() as folder:
            data_dir = Path(folder) / "must-not-exist"
            configured = "postgresql+psycopg2://user:password@example.invalid/disasters"

            url = resolve_database_url(
                {"VERCEL": "1", "DATABASE_URL": configured}, data_dir
            )

            self.assertEqual(url, configured)
            self.assertFalse(data_dir.exists())


if __name__ == "__main__":
    unittest.main()
