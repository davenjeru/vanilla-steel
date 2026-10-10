"""What happens when an enquiry arrives: store it, then let the agent handle it.

Synchronous. The agent's tools move the enquiry through its statuses; the
agent run (every LLM turn and tool call) is stored for evals and audits.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from steel.agent.runner import run_agent
from steel.agent.tools import ToolContext
from steel.domain.enquiry import Enquiry
from steel.domain.fingerprint import enquiry_fingerprint
from steel.domain.status import start_history
from steel.extraction.attachments import Attachment
from steel.llm.client import LLMClient
from steel.storage import agent_runs, enquiries


class EnquirySubmission(BaseModel):
    sender: str
    subject: str
    body: str
    attachments: list[Attachment] = []


@dataclass(frozen=True)
class Pipeline:
    """Everything a pipeline step needs, injected so tests control the LLM, the clock and IDs."""

    connection: sqlite3.Connection
    llm: LLMClient
    extraction_model: str
    agent_model: str
    quote_model: str
    clock: Callable[[], datetime]
    new_enquiry_id: Callable[[], str]
    new_rfq_id: Callable[[], str]
    new_quote_id: Callable[[], str]


def receive_enquiry(pipeline: Pipeline, tenant_id: str, submission: EnquirySubmission) -> Enquiry:
    received_at = pipeline.clock()
    enquiry = Enquiry(
        id=pipeline.new_enquiry_id(),
        tenant_id=tenant_id,
        sender=submission.sender,
        subject=submission.subject,
        body=submission.body,
        fingerprint=enquiry_fingerprint(submission.subject, submission.body, submission.attachments),
        status_history=start_history(received_at),
        created_at=received_at,
    )
    enquiries.insert_enquiry(pipeline.connection, enquiry, submission.attachments)

    context = ToolContext(
        tenant_id=tenant_id,
        enquiry_id=enquiry.id,
        connection=pipeline.connection,
        llm=pipeline.llm,
        extraction_model=pipeline.extraction_model,
        quote_model=pipeline.quote_model,
        clock=pipeline.clock,
        new_rfq_id=pipeline.new_rfq_id,
        new_quote_id=pipeline.new_quote_id,
    )
    run = run_agent(context, pipeline.agent_model)
    agent_runs.save_agent_run(pipeline.connection, tenant_id, run)
    return context.load_enquiry()
