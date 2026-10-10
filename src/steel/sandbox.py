"""Handle one enquiry in a throwaway database, with predictable IDs.

For trying the agent and for agent evals: every run starts from the same seed
stock with IDs enq_0001 and rfq_0001, so every tool result is identical, and a
recorded run replays exactly.
"""

import itertools
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from steel.agent.trace import AgentRun
from steel.domain.enquiry import Enquiry, Rfq
from steel.extraction.attachments import Attachment
from steel.llm.client import LLMClient
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import agent_runs, rfqs
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database

SANDBOX_TENANT = "tnt_bauer"
SANDBOX_TIME = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)


class SandboxResult(BaseModel):
    enquiry: Enquiry
    run: AgentRun
    rfq: Rfq | None


def handle_in_sandbox(
    subject: str, body: str, attachments: list[Attachment], llm: LLMClient, settings: Settings
) -> SandboxResult:
    with tempfile.TemporaryDirectory() as directory:
        connection = connect(Path(directory) / "sandbox.db")
        create_schema(connection)
        seed_database(connection)
        enquiry_numbers, rfq_numbers = itertools.count(1), itertools.count(1)
        pipeline = Pipeline(
            connection=connection,
            llm=llm,
            extraction_model=settings.extraction_model,
            agent_model=settings.agent_model,
            clock=lambda: SANDBOX_TIME,
            new_enquiry_id=lambda: f"enq_{next(enquiry_numbers):04d}",
            new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
        )
        submission = EnquirySubmission(sender="buyer@example.com", subject=subject, body=body, attachments=attachments)
        enquiry = receive_enquiry(pipeline, SANDBOX_TENANT, submission)
        run = agent_runs.get_agent_run(connection, SANDBOX_TENANT, enquiry.id)
        rfq = rfqs.get_rfq_for_enquiry(connection, SANDBOX_TENANT, enquiry.id)
        connection.close()
    if run is None:
        raise RuntimeError(f"no agent run stored for {enquiry.id}")
    return SandboxResult(enquiry=enquiry, run=run, rfq=rfq)
