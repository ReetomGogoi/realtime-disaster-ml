"""Database URL selection without opening a database connection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


_VERCEL_POSTGRESQL_SCHEMES = frozenset({"postgresql", "postgresql+psycopg2"})


def resolve_database_url(environment: Mapping[str, str], data_dir: Path) -> str:
    """Select hosted PostgreSQL on Vercel or the existing local SQLite file."""
    configured_url = environment.get("DATABASE_URL", "").strip()
    is_vercel = environment.get("VERCEL") == "1"

    if is_vercel:
        if not configured_url:
            raise RuntimeError(
                "DATABASE_URL must be configured with a persistent PostgreSQL "
                "database for Vercel deployments."
            )
        scheme = configured_url.partition(":")[0].lower()
        if scheme not in _VERCEL_POSTGRESQL_SCHEMES:
            raise RuntimeError(
                "Vercel deployments require a PostgreSQL DATABASE_URL; "
                "SQLite and other local database URLs are not supported."
            )

    if configured_url:
        return configured_url

    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{data_dir / 'disasters.db'}"
