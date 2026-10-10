import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(database_path: Path | str) -> sqlite3.Connection:
    """Open a connection with named-column rows and foreign keys enforced.

    check_same_thread is off because FastAPI may open the connection in one
    worker thread and use it in another. Each request gets its own connection,
    so it is never shared concurrently.
    """
    connection = sqlite3.connect(database_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_PATH.read_text())
