"""Domain exceptions and the FastAPI handlers that turn them into responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ClaimGuardError(Exception):
    """Base class for everything this application raises deliberately."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(ClaimGuardError):
    status_code = 404
    code = "not_found"


class ValidationError(ClaimGuardError):
    status_code = 422
    code = "validation_error"


class AgentFailure(ClaimGuardError):
    """Raised when an agent cannot produce a usable result.

    The orchestrator catches these: one broken agent degrades the
    investigation, it does not kill it.
    """

    status_code = 502
    code = "agent_failure"


class ExternalServiceError(ClaimGuardError):
    status_code = 503
    code = "external_service_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ClaimGuardError)
    async def _handle(_: Request, exc: ClaimGuardError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )
