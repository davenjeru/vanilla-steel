"""Turn enquiry attachments into text the extractor can read.

Spreadsheets are converted by code, not by the LLM, so a table reaches the
model with its rows and columns intact. Every attachment becomes:

    Attachment order.xlsx, sheet Order:
    grade | thickness_mm | quantity_t
    S235JR | 3 | 25
"""

import csv
import io
import zipfile
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import PurePath

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from pydantic import BaseModel

# Excel exports in Germany and Poland are often Windows-1252 rather than UTF-8.
TEXT_ENCODINGS = ("utf-8-sig", "cp1252")
CSV_DELIMITERS = ";,\t"
CSV_CONTENT_TYPE = "text/csv"
XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

Cell = str | int | float | date | datetime | None


class Attachment(BaseModel):
    filename: str
    content_type: str
    content: bytes


class UnsupportedAttachment(ValueError):
    pass


def attachment_as_text(attachment: Attachment) -> str:
    extension = PurePath(attachment.filename).suffix.lower()
    if extension == ".csv" or attachment.content_type == CSV_CONTENT_TYPE:
        return _csv_as_text(attachment)
    if extension == ".xlsx" or attachment.content_type == XLSX_CONTENT_TYPE:
        return _xlsx_as_text(attachment)
    raise UnsupportedAttachment(f"{attachment.filename}: {extension or attachment.content_type} is not supported yet")


def _csv_as_text(attachment: Attachment) -> str:
    text = _decode(attachment.content)
    rows = csv.reader(io.StringIO(text), _csv_dialect(text))
    return _table(f"Attachment {attachment.filename}", rows)


def _csv_dialect(text: str) -> type[csv.Dialect] | csv.Dialect:
    """Detect ; , or tab. A single-column file has no delimiter to detect, and any dialect reads it."""
    try:
        return csv.Sniffer().sniff(text[:2048], delimiters=CSV_DELIMITERS)
    except csv.Error:
        return csv.excel


def _xlsx_as_text(attachment: Attachment) -> str:
    """Every sheet, with cached values: a formula shows its last computed result, not the formula."""
    try:
        workbook = load_workbook(io.BytesIO(attachment.content), read_only=True, data_only=True)
    except (zipfile.BadZipFile, InvalidFileException, KeyError) as error:
        raise UnsupportedAttachment(f"{attachment.filename}: not a readable .xlsx file") from error
    try:
        tables = [
            _table(f"Attachment {attachment.filename}, sheet {sheet.title}", _cells_as_text(sheet.values))
            for sheet in workbook.worksheets
        ]
    finally:
        workbook.close()
    return "\n\n".join(tables)


def _cells_as_text(rows: Iterable[tuple[Cell, ...]]) -> Iterable[list[str]]:
    for row in rows:
        yield [_cell_as_text(cell) for cell in row]


def _cell_as_text(cell: Cell) -> str:
    if cell is None:
        return ""
    if isinstance(cell, datetime):
        return cell.date().isoformat() if cell.time() == datetime.min.time() else cell.isoformat()
    if isinstance(cell, date):
        return cell.isoformat()
    if isinstance(cell, float) and cell.is_integer():
        return str(int(cell))  # Excel stores 3 as 3.0; show it as written
    return str(cell)


def _table(heading: str, rows: Iterable[list[str]]) -> str:
    """One line per non-empty row, cells separated by |."""
    lines = [" | ".join(cell.strip() for cell in row) for row in rows if any(cell.strip() for cell in row)]
    return heading + ":\n" + "\n".join(lines)


def _decode(content: bytes) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedAttachment("attachment is neither UTF-8 nor Windows-1252 text")
