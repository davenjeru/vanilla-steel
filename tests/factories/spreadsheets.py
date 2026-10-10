import io
from typing import Any

from openpyxl import Workbook

from steel.extraction.attachments import XLSX_CONTENT_TYPE, Attachment


def xlsx_bytes(sheets: dict[str, list[list[Any]]]) -> bytes:
    """A real .xlsx file with the given sheets, each a list of rows."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def xlsx_attachment(sheets: dict[str, list[list[Any]]], filename: str = "order.xlsx") -> Attachment:
    return Attachment(filename=filename, content_type=XLSX_CONTENT_TYPE, content=xlsx_bytes(sheets))
