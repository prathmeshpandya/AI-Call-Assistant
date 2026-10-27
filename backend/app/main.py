"""FastAPI entry point.

Run (from backend/):  .venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
or simply:           .venv\\Scripts\\python -m app.main
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as ws_router
from app.config import Settings, get_settings
from app.core.session_manager import SessionManager
from app.logging_config import configure_logging
from app.services import Services

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = services or await asyncio.to_thread(Services.load, settings)  # model loading is blocking
        app.state.services = svc
        app.state.sessions = SessionManager(svc, settings.max_sessions)
        warmup = asyncio.create_task(svc.warmup()) if settings.warmup else None
        log.info("Backend ready", extra={"host": settings.backend_host, "port": settings.backend_port,
                                         "speech_ready": svc.speech_ready})
        try:
            yield
        finally:
            if warmup and not warmup.done():
                warmup.cancel()
            await app.state.sessions.close_all()
            await svc.aclose()

    app = FastAPI(title="SLM Voice Assistant", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_methods=["GET"], allow_headers=["*"])
    app.include_router(ws_router)

    @app.get("/api/health")
    async def health() -> dict:
        svc: Services = app.state.services
        llm = await svc.llm.health()
        return {
            "status": "ok" if svc.speech_ready and llm.ok else "degraded",
            "components": {
                "stt": svc.errors.get("stt", "ok"),
                "tts": svc.errors.get("tts", "ok"),
                "vad": svc.errors.get("vad", "ok"),
                "llm": "ok" if llm.ok else llm.detail,
            },
            "sessions": app.state.sessions.count,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.backend_host, port=s.backend_port, log_config=None)
