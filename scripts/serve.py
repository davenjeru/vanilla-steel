"""Serve the steel procurement API with Claude behind it, for a hands-on demo.

Usage: uv run --env-file .env python -m scripts.serve [--port 8000]        (make serve)

Start from a fresh database (make seed) so a demo's first enquiry is not a duplicate of yesterday's.
The three tenants' development keys are in steel/storage/seed.py; send one in the X-API-Key header.
"""

import argparse

import uvicorn

from steel.api.app import create_app
from steel.llm.anthropic_client import AnthropicClient
from steel.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    app = create_app(Settings(), AnthropicClient.from_environment())
    uvicorn.run(app, host="127.0.0.1", port=arguments.port)


if __name__ == "__main__":
    main()
