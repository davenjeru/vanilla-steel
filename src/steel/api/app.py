from fastapi import FastAPI

from steel.settings import Settings
from steel.storage.db import connect


def create_app(settings: Settings) -> FastAPI:
    """Build the API. Tests pass their own settings (and, from Slice 1, their own LLM client)."""
    app = FastAPI(title="Steel procurement pipeline")
    app.state.settings = settings

    @app.get("/health")
    def health() -> dict[str, str | int]:
        connection = connect(settings.database_path)
        try:
            stock_items = connection.execute("SELECT COUNT(*) FROM stock_items").fetchone()[0]
        finally:
            connection.close()
        return {"status": "ok", "stock_items": stock_items}

    return app
