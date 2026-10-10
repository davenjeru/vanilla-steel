"""Request-scoped dependencies: a database connection and the calling tenant."""

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from steel.domain.models import Tenant
from steel.storage import api_keys
from steel.storage.db import connect


def database_connection(request: Request) -> Iterator[sqlite3.Connection]:
    connection = connect(request.app.state.settings.database_path)
    try:
        yield connection
    finally:
        connection.close()


Connection = Annotated[sqlite3.Connection, Depends(database_connection)]


def current_tenant(connection: Connection, x_api_key: Annotated[str | None, Header()] = None) -> Tenant:
    """The caller's tenant comes from its API key alone; no header can name a tenant directly.

    An unknown key gets the same answer as a missing one, so the API reveals nothing about which keys exist.
    """
    tenant = api_keys.find_tenant_by_api_key(connection, x_api_key) if x_api_key else None
    if tenant is None:
        raise api_error(401, "INVALID_API_KEY", "send your API key in the X-API-Key header")
    return tenant


CurrentTenant = Annotated[Tenant, Depends(current_tenant)]


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Errors carry a stable code that clients and contract tests can rely on."""
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})
