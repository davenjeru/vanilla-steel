from fastapi import APIRouter

from steel.api.deps import Connection, CurrentTenant, api_error
from steel.api.schemas import BuyerQuote
from steel.storage import quotes

router = APIRouter(prefix="/v1/quotes", tags=["quotes"])


@router.get("/{quote_id}")
def read_quote(quote_id: str, tenant: CurrentTenant, connection: Connection) -> BuyerQuote:
    """Another tenant's quote is reported as not found, never as forbidden: its existence stays hidden."""
    quote = quotes.get_quote(connection, tenant.id, quote_id)
    if quote is None:
        raise api_error(404, "QUOTE_NOT_FOUND", f"no quote {quote_id}")
    return BuyerQuote.from_quote(quote)
