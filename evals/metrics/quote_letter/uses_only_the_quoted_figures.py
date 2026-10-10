import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.quote_letter.letter_case import read_letter_case
from steel.domain.quote import QUOTE_VALID_WORKING_DAYS, PricedQuote

# Dates, checked before numbers so that 15.10.2026 is not read as the number 15.10.
DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("year", "month", "day")),
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), ("day", "month", "year")),
]
# A number not glued to letters, so the digits in S355J2 or DX51D+Z275 are names, not figures.
# Either grouped in thousands (21,030.00 / 21.030,00 / 21 030,00) or plain (701, 15.5).
NUMBER = re.compile(
    r"(?<![\w.,])(?:\d{1,3}(?:[.,   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\w])",
)


class UsesOnlyTheQuotedFigures(BaseMetric):
    """Every number and date in the letter comes from the quote: quantity, dimensions, price, total,
    valid-until date (and the 3 working days behind it). Zero tolerance.

    Prices are deterministic (D5); the letter only wraps them in words. A figure the quote never gave
    (a rounded total, a "special price", another date) is an offer the buyer could hold the platform to.
    Deterministic: English, German and Polish number formats are all accepted.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Uses only the quoted figures"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        case = read_letter_case(test_case)
        quote = case.writer_input.quote
        wrong_dates, text_without_dates = _check_dates(case.letter, quote.valid_until)
        numbers = NUMBER.findall(text_without_dates)
        wrong_numbers = [token for token in numbers if not _is_quoted(token, _quoted_figures(quote))]
        unquoted = wrong_dates + wrong_numbers
        self.success = not unquoted
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(unquoted, figure_count=len(numbers) + len(wrong_dates))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _quoted_figures(quote: PricedQuote) -> set[Decimal]:
    figures = {Decimal(QUOTE_VALID_WORKING_DAYS)}
    figures |= {Decimal(quote.valid_until.year), Decimal(quote.valid_until.month), Decimal(quote.valid_until.day)}
    for line in quote.lines:
        measures = [line.quantity_t, line.thickness_mm, line.width_mm, line.length_mm]
        figures |= {Decimal(str(value)) for value in measures if value is not None}
        figures |= {line.price_per_tonne_eur, line.total_eur}
    return figures


def _check_dates(letter: str, valid_until: date) -> tuple[list[str], str]:
    """Dates other than the valid-until date, and the letter with every date removed."""
    wrong: list[str] = []
    for pattern, order in DATE_PATTERNS:
        for match in pattern.finditer(letter):
            parts = dict(zip(order, (int(group) for group in match.groups()), strict=True))
            if (parts["year"], parts["month"], parts["day"]) != (valid_until.year, valid_until.month, valid_until.day):
                wrong.append(match.group(0))
        letter = pattern.sub(" ", letter)
    return wrong, letter


def _is_quoted(token: str, quoted: set[Decimal]) -> bool:
    """Accepted if either reading matches: English (1,234.5) or German and Polish (1.234,5 / 1 234,5)."""
    return any(value in quoted for value in _readings(token))


def _readings(token: str) -> list[Decimal]:
    without_spaces = re.sub(r"[   ]", "", token)
    candidates = [without_spaces.replace(",", ""), without_spaces.replace(".", "").replace(",", ".")]
    readings = []
    for candidate in candidates:
        try:
            readings.append(Decimal(candidate))
        except InvalidOperation:
            continue
    return readings


def _reason(unquoted: list[str], figure_count: int) -> str:
    if unquoted:
        return f"Figures not in the quote: {', '.join(unquoted)}"
    if figure_count == 0:
        return "The letter states no figures"
    return f"All {figure_count} figures come from the quote"
