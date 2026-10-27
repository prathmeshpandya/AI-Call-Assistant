"""Registry of live voice sessions with a concurrency cap and guaranteed cleanup."""

from __future__ import annotations

import asyncio
import logging
import uuid

from app.core.transport import ClientTransport
from app.core.voice_session import VoiceSession
from app.services import Services

log = logging.getLogger(__name__)


class TooManySessionsError(RuntimeError):
    pass


class SessionManager:
    def __init__(self, services: Services, max_sessions: int) -> None:
        self._services = services
        self._max = max_sessions
        self._sessions: dict[str, VoiceSession] = {}
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self._sessions)

    async def open(self, transport: ClientTransport) -> VoiceSession:
        async with self._lock:
            if len(self._sessions) >= self._max:
                raise TooManySessionsError(f"Maximum of {self._max} concurrent sessions reached")
            session = VoiceSession(uuid.uuid4().hex[:12], self._services, transport)
            self._sessions[session.id] = session
        log.info("Session opened", extra={"session": session.id, "active": len(self._sessions)})
        await session.start()
        return session

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close()

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            await self.close(session_id)
