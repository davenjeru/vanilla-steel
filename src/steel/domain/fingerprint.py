"""A fingerprint that is the same for a resent or forwarded enquiry.

Exact matching after normalisation: case, spacing and reply/forward prefixes
(Re:, Fwd:, and German AW:, WG:) are ignored. Reworded enquiries are a
different fingerprint; near-duplicate detection is Slice 3.
"""

import hashlib
import re

SUBJECT_PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|wg)\s*:\s*)+", flags=re.IGNORECASE)


def enquiry_fingerprint(subject: str, body: str) -> str:
    normalised = f"{_normalise_subject(subject)}\n{_normalise_text(body)}"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _normalise_subject(subject: str) -> str:
    return _normalise_text(SUBJECT_PREFIX.sub("", subject))


def _normalise_text(text: str) -> str:
    return " ".join(text.lower().split())
