"""A fingerprint that is the same for a resent or forwarded enquiry.

Exact matching after normalisation: case, spacing and reply/forward prefixes
(Re:, Fwd:, and German AW:, WG:) are ignored. Attachments count by content,
not by file name or order: "weekly order, see attached" with a new spreadsheet
is a new order (finding F-005), while a forward that renames the file is not.
Reworded enquiries get a different fingerprint; the order-based check catches
those (D36).
"""

import hashlib
import re

from steel.extraction.attachments import Attachment

SUBJECT_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|wg)\s*:\s*)+", flags=re.IGNORECASE)


def enquiry_fingerprint(subject: str, body: str, attachments: list[Attachment]) -> str:
    attachment_hashes = sorted(_content_hash(attachment.content) for attachment in attachments)
    normalised = "\n".join([_normalise_subject(subject), _normalise_text(body), *attachment_hashes])
    return _content_hash(normalised.encode("utf-8"))


def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalise_subject(subject: str) -> str:
    return _normalise_text(SUBJECT_PREFIX.sub("", subject))


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())
