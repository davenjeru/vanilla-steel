import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from fastapi import FastAPI

from steel.api import enquiries, quotes
from steel.llm.client import LLMClient
from steel.settings import Settings
from steel.storage.db import connect


def utc_now() -> datetime:
    return datetime.now(UTC)


def random_enquiry_id() -> str:
    return f"enq_{uuid.uuid4().hex[:12]}"


def random_rfq_id() -> str:
    return f"rfq_{uuid.uuid4().hex[:12]}"


def random_quote_id() -> str:
    return f"quo_{uuid.uuid4().hex[:12]}"


def create_app(
    settings: Settings,
    llm: LLMClient,
    clock: Callable[[], datetime] = utc_now,
    new_enquiry_id: Callable[[], str] = random_enquiry_id,
    new_rfq_id: Callable[[], str] = random_rfq_id,
    new_quote_id: Callable[[], str] = random_quote_id,
) -> FastAPI:
    """Build the API. Tests inject their own LLM, clock and ID generator so results are deterministic."""
    app = FastAPI(title="Steel procurement pipeline")
    app.state.settings = settings
    app.state.llm = llm
    app.state.clock = clock
    app.state.new_enquiry_id = new_enquiry_id
    app.state.new_rfq_id = new_rfq_id
    app.state.new_quote_id = new_quote_id
    app.include_router(enquiries.router)
    app.include_router(quotes.router)

    @app.get("/health")
    def health() -> dict[str, str | int]:
        connection = connect(settings.database_path)
        try:
            stock_items = connection.execute("SELECT COUNT(*) FROM stock_items").fetchone()[0]
        finally:
            connection.close()
        return {"status": "ok", "stock_items": stock_items}

    return app
