"""Spreadsheet attachments must reach the extractor with rows and columns intact."""

from datetime import date

import pytest

from steel.extraction.attachments import XLSX_CONTENT_TYPE, Attachment, UnsupportedAttachment, attachment_as_text
from tests.factories.spreadsheets import xlsx_attachment


def csv_attachment(content: bytes, filename: str = "order.csv") -> Attachment:
    return Attachment(filename=filename, content_type="text/csv", content=content)


def test_comma_separated_rows_become_a_readable_table() -> None:
    text = attachment_as_text(csv_attachment(b"grade,thickness_mm,quantity\nS235JR,3,25\n"))

    assert text == "Attachment order.csv:\ngrade | thickness_mm | quantity\nS235JR | 3 | 25"


def test_semicolon_separated_rows_are_split_correctly() -> None:
    """German and Polish Excel exports use semicolons, because the comma is the decimal separator."""
    text = attachment_as_text(csv_attachment("Güte;Dicke;Menge\nS235JR;2,5;25\n".encode()))

    assert text.splitlines()[2] == "S235JR | 2,5 | 25"


def test_windows_1252_export_is_decoded() -> None:
    text = attachment_as_text(csv_attachment("Güte;Menge\nS235JR;25\n".encode("cp1252")))

    assert "Güte" in text


def test_excel_byte_order_mark_is_not_part_of_the_first_header() -> None:
    text = attachment_as_text(csv_attachment("grade,quantity\nS235JR,25\n".encode("utf-8-sig")))

    assert text.splitlines()[1].startswith("grade")


def test_empty_rows_are_dropped() -> None:
    text = attachment_as_text(csv_attachment(b"grade,quantity\n,\nS235JR,25\n"))

    assert text.splitlines()[1:] == ["grade | quantity", "S235JR | 25"]


def test_unsupported_attachment_is_rejected_with_its_name() -> None:
    pdf = Attachment(filename="drawing.pdf", content_type="application/pdf", content=b"%PDF-1.7")

    with pytest.raises(UnsupportedAttachment, match="drawing.pdf"):
        attachment_as_text(pdf)


def test_single_column_csv_is_read_instead_of_crashing() -> None:
    """Regression: csv.Sniffer cannot find a delimiter in one column and raised, failing the whole request."""
    text = attachment_as_text(csv_attachment(b"grade\nS235JR\nS355J2\n"))

    assert text.splitlines()[1:] == ["grade", "S235JR", "S355J2"]


# Excel


def test_excel_rows_become_a_readable_table_named_by_sheet() -> None:
    attachment = xlsx_attachment({"Order": [["grade", "thickness_mm", "quantity_t"], ["S235JR", 3, 25]]})

    text = attachment_as_text(attachment)

    assert text == "Attachment order.xlsx, sheet Order:\ngrade | thickness_mm | quantity_t\nS235JR | 3 | 25"


def test_excel_whole_numbers_show_without_a_decimal_point() -> None:
    """Excel stores 3 as 3.0; the extractor should see what the customer typed."""
    text = attachment_as_text(xlsx_attachment({"Order": [["thickness_mm"], [3.0], [2.5]]}))

    assert text.splitlines()[2:] == ["3", "2.5"]


def test_excel_dates_become_iso_dates() -> None:
    text = attachment_as_text(xlsx_attachment({"Order": [["delivery"], [date(2026, 11, 30)]]}))

    assert text.splitlines()[2] == "2026-11-30"


def test_every_excel_sheet_is_included() -> None:
    attachment = xlsx_attachment({"Sheets": [["S235JR", 3]], "Coils": [["S355MC", 4]]})

    text = attachment_as_text(attachment)

    assert "sheet Sheets:\nS235JR | 3" in text and "sheet Coils:\nS355MC | 4" in text


def test_empty_excel_rows_are_dropped() -> None:
    text = attachment_as_text(xlsx_attachment({"Order": [["grade"], [None], ["S235JR"]]}))

    assert text.splitlines()[1:] == ["grade", "S235JR"]


def test_excel_is_recognised_by_content_type_without_an_extension() -> None:
    attachment = xlsx_attachment({"Order": [["S235JR"]]}, filename="order")

    assert attachment_as_text(attachment) == "Attachment order, sheet Order:\nS235JR"


def test_corrupt_excel_file_is_rejected_instead_of_crashing() -> None:
    corrupt = Attachment(filename="order.xlsx", content_type=XLSX_CONTENT_TYPE, content=b"not a zip file")

    with pytest.raises(UnsupportedAttachment, match="not a readable .xlsx file"):
        attachment_as_text(corrupt)


def test_legacy_xls_is_not_supported() -> None:
    """openpyxl reads only .xlsx; the old binary .xls format needs another library."""
    legacy = Attachment(filename="order.xls", content_type="application/vnd.ms-excel", content=b"\xd0\xcf\x11\xe0")

    with pytest.raises(UnsupportedAttachment, match=r"\.xls is not supported"):
        attachment_as_text(legacy)
