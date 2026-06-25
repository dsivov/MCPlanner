import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level above backend/)
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env", override=False)


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    MODEL_SELECTION: str = os.getenv("MODEL_SELECTION", "gpt-4o")
    MODEL_ROLLOUT: str = os.getenv("MODEL_ROLLOUT", "gpt-4o-mini")
    MODEL_USER_SIM: str = os.getenv("MODEL_USER_SIM", "gpt-4o-mini")
    MODEL_STATE: str = os.getenv("MODEL_STATE", "gpt-4o-mini")
    MODEL_BUILDER: str = os.getenv("MODEL_BUILDER", "gpt-4o")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite+aiosqlite:///{ROOT / 'data' / 'planner.db'}"
    )
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
    # Avatar / GPT-Realtime (voice). Defaults match the standalone Avatar project.
    REALTIME_MODEL: str = os.getenv("REALTIME_MODEL", "gpt-realtime-2")
    REALTIME_VOICE: str = os.getenv("REALTIME_VOICE", "marin")
    # PASTE-style speculative scheduler: max concurrent background (pondering /
    # instruction-prefetch) LLM calls. Bounds wasted work on mispredictions and keeps the
    # live turn from queueing behind speculative load.
    SPECULATIVE_BUDGET: int = int(os.getenv("SPECULATIVE_BUDGET", "4"))


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "sops").mkdir(parents=True, exist_ok=True)
