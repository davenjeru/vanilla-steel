"""Agent run persistence. Tenant-owned and internal: the trace can name stock lots."""

import sqlite3

from steel.agent.trace import AgentRun


def save_agent_run(connection: sqlite3.Connection, tenant_id: str, run: AgentRun) -> None:
    with connection:
        connection.execute(
            "INSERT OR REPLACE INTO agent_runs (enquiry_id, tenant_id, run) VALUES (?, ?, ?)",
            (run.enquiry_id, tenant_id, run.model_dump_json()),
        )


def get_agent_run(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> AgentRun | None:
    row = connection.execute(
        "SELECT run FROM agent_runs WHERE enquiry_id = ? AND tenant_id = ?", (enquiry_id, tenant_id)
    ).fetchone()
    return AgentRun.model_validate_json(row["run"]) if row else None
