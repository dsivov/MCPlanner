from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import init_db
from .routes.sop import router as sop_router
from .routes.chat import router as chat_router
from .routes.experiments import router as experiments_router
from .routes.learn import router as learn_router
from .routes.context_graph import router as context_graph_router
from .routes.avatar import router as avatar_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="PCA Planner POC", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Allow any localhost port — Vite picks 5173/5174/… depending on what's free.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sop_router)
app.include_router(chat_router)
app.include_router(experiments_router)
app.include_router(learn_router)
app.include_router(context_graph_router)
app.include_router(avatar_router)


@app.get("/")
async def root() -> dict:
    return {
        "service": "PCA Planner POC backend",
        "ui": "http://127.0.0.1:5173",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "models": {
            "selection": settings.MODEL_SELECTION,
            "rollout": settings.MODEL_ROLLOUT,
            "user_sim": settings.MODEL_USER_SIM,
            "state": settings.MODEL_STATE,
            "builder": settings.MODEL_BUILDER,
        },
    }
