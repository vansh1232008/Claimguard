"""Central configuration for ClaimGuard.

Everything here has an offline-safe default. The platform boots and runs a full
investigation with zero API keys and zero external services; adding credentials
to `.env` upgrades each subsystem to its production backend.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:  # pydantic-settings is optional at import time so the module never hard-fails
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover - fallback for minimal environments
    from pydantic import BaseModel as BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore
    _HAS_PYDANTIC_SETTINGS = False


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    """Runtime settings, read from environment variables / .env."""

    # --- app ---------------------------------------------------------------
    app_name: str = "ClaimGuard"
    app_env: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- persistence -------------------------------------------------------
    # Default is SQLite so the project runs with no database server.
    database_url: str = f"sqlite:///{(REPO_ROOT / 'claimguard.db').as_posix()}"

    # --- LLM ---------------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 2048

    # --- vector store (policy RAG) ----------------------------------------
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "claimguard_policies"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4

    # --- graph store -------------------------------------------------------
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    # --- ml artefacts ------------------------------------------------------
    artifact_dir: str = str(ARTIFACT_DIR)
    xgboost_model_file: str = "fraud_xgboost.json"
    graphsage_embedding_file: str = "graphsage_embeddings.npz"
    feature_metadata_file: str = "feature_metadata.json"

    # --- decision policy ---------------------------------------------------
    auto_approve_below: float = 0.25
    auto_reject_above: float = 0.85
    high_value_claim_threshold: float = 15000.0

    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(
            env_file=(REPO_ROOT / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
        )

    # --- derived helpers ---------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def artifacts(self) -> Path:
        p = Path(self.artifact_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads(self) -> Path:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return UPLOAD_DIR

    @property
    def policy_corpus_dir(self) -> Path:
        return DATA_DIR / "policies"

    @property
    def llm_mode(self) -> str:
        return "gemini" if self.gemini_api_key else "offline-stub"

    @property
    def vector_mode(self) -> str:
        return "qdrant" if self.qdrant_url else "in-memory"

    @property
    def graph_mode(self) -> str:
        return "neo4j" if self.neo4j_uri else "in-memory"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    if not _HAS_PYDANTIC_SETTINGS:  # pragma: no cover
        # Minimal manual env loading when pydantic-settings is unavailable.
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        kwargs = {}
        for field in Settings.model_fields:  # type: ignore[attr-defined]
            val = os.environ.get(field.upper())
            if val is not None:
                kwargs[field] = val
        return Settings(**kwargs)
    return Settings()


settings = get_settings()
