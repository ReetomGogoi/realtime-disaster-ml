from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.session import engine
from app.database.base import Base
import app.database.models  # noqa: F401 — registers all models with Base
from app.api.routes.events import router as events_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup (safe: does not drop existing tables)."""
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Realtime Disaster ML API",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server to call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(events_router, prefix="", tags=["events"])


# ---------------------------------------------------------------------------
# Root-level endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "realtime-disaster-ml-api",
        "version": "0.1.0",
    }
