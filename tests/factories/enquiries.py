import base64
from typing import Any

from steel.storage.seed import DEVELOPMENT_API_KEYS

# Each tenant calls the API with its own key; the key alone decides the tenant.
BAUER = {"X-API-Key": DEVELOPMENT_API_KEYS["tnt_bauer"]}
KOWALSKI = {"X-API-Key": DEVELOPMENT_API_KEYS["tnt_kowalski"]}


def enquiry_payload(**overrides: Any) -> dict[str, Any]:
    """The standard test enquiry as a POST /v1/enquiries body."""
    payload: dict[str, Any] = {
        "sender": "anna@bauer-metallbau.example",
        "subject": "Enquiry St37 sheets",
        "body": "Hi, we need 25 t of St37-2 hot rolled sheet 3 x 1500 x 3000 mm, DAP Poznań, by 30 November 2026.",
        "attachments": [],
    }
    return payload | overrides


def attachment_payload(filename: str, content: bytes, content_type: str = "text/csv") -> dict[str, str]:
    return {
        "filename": filename,
        "content_type": content_type,
        "content_base64": base64.b64encode(content).decode("ascii"),
    }
