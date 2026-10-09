import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# Database URL
# ---------------------------------------------------------------------------
# Default: SQLite stored at backend/data/disasters.db.
# Override DATABASE_URL env var to switch to PostgreSQL without code changes.
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parents[3]  # project root / backend/
_DATA_DIR = _BASE_DIR / "backend" / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_DATA_DIR / 'disasters.db'}",
)

# SQLite-specific arg; ignored when using PostgreSQL
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
