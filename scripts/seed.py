"""Create a fresh, seeded SQLite database.

Usage: uv run python scripts/seed.py [--database data/steel.db]
"""

import argparse
from pathlib import Path

from steel.settings import Settings
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Settings().database_path)
    database_path: Path = parser.parse_args().database

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)
    connection = connect(database_path)
    create_schema(connection)
    seed_database(connection)
    connection.close()
    print(f"Seeded {database_path}")


if __name__ == "__main__":
    main()
