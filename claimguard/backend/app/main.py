"""ClaimGuard API entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger
from app.database import init_db, session_scope

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s (%s)", settings.app_name, settings.app_env)
    init_db()

    # Warm the swappable subsystems so the first request is not the slow one.
    from app.services.graph_sync import refresh_graph
    from app.services.llm import get_llm
    from app.services.ml import get_scorer
    from app.services.vectorstore import get_vector_store

    get_llm()
    get_vector_store()
    get_scorer()
    try:
        with session_scope() as db:
            refresh_graph(db)
    except Exception as exc:  # empty database on first boot
        logger.info("Graph warm-up skipped: %s", exc)

    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ClaimGuard API",
    version="1.0.0",
    description=(
        "Autonomous insurance claims investigation platform. A seven-agent pipeline "
        "(intake, documents, damage, policy RAG, anomaly model, network analysis, "
        "adjudication) that returns an explainable approve / review / reject "
        "recommendation with the evidence behind it."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "api": settings.api_prefix,
        "modes": {
            "llm": settings.llm_mode,
            "vector_store": settings.vector_mode,
            "graph": settings.graph_mode,
        },
    }
