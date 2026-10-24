import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.session import engine
from app.database.base import Base
import app.database.models  # noqa: F401 — registers all models with Base
from app.api.routes.events import router as events_router
from app.ml_inference import load_model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup (safe: does not drop existing tables)."""
    Base.metadata.create_all(bind=engine)
    app.state.ml_bundle = load_model_bundle()
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

if os.getenv("ENABLE_LOCAL_ML_DEMO") == "1" and not os.getenv("VERCEL"):
    from app.api.routes.local_demo import router as local_demo_router
    app.include_router(local_demo_router, tags=["local demonstration"])


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
