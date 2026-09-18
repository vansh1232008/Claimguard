from fastapi import APIRouter

from app.api.routes import claims, graph, investigations, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(claims.router)
api_router.include_router(investigations.router)
api_router.include_router(graph.router)

__all__ = ["api_router"]
