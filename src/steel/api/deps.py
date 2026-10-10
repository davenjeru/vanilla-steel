"""Request-scoped dependencies: a database connection and the calling tenant."""

import sqlite3
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from steel.domain.models import Tenant
from steel.storage import repositories
from steel.storage.db import connect


def database_connection(request: Request) -> Iterator[sqlite3.Connection]:
    connection = connect(request.app.state.settings.database_path)
    try:
        yield connection
    finally:
        connection.close()


Connection = Annotated[sqlite3.Connection, Depends(database_connection)]


def current_tenant(connection: Connection, x_tenant_id: Annotated[str | None, Header()] = None) -> Tenant:
    """Temporary: Slice 3 replaces this header with per-tenant API keys.

    An unknown tenant gets the same answer as a missing one, so the API
    does not reveal which tenant IDs exist.
    """
    tenant = repositories.get_tenant(connection, x_tenant_id) if x_tenant_id else None
    if tenant is None:
        raise api_error(401, "UNKNOWN_TENANT", "send a known tenant ID in the X-Tenant-ID header")
    return tenant


CurrentTenant = Annotated[Tenant, Depends(current_tenant)]


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Errors carry a stable code that clients and contract tests can rely on."""
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})
