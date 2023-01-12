from pathlib import Path

import environs

PROJECT_DIR = Path(__file__).parent.parent.resolve()
BACKEND_DIR = PROJECT_DIR / "backend"
MODEL_DIR = PROJECT_DIR / "backend/models"

env = environs.Env()
env.read_env(BACKEND_DIR / ".env", recurse=False)

HOST = env.str("HOST", "127.0.0.1")
PORT = env.int("PORT", 8000)
