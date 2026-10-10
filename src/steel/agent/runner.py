"""The agent loop: Claude decides which tool to call next; code runs the tool and records every step.

Two safety nets are code, not prompt: the loop stops after MAX_TURNS, and a
run that ends without a decision is escalated, so an enquiry never silently
stalls.
"""

import json
import time
from pathlib import Path
from typing import Any

from steel.agent.tools import TOOLS, ToolContext, ToolError, escalate_enquiry
from steel.agent.trace import AgentRun, ModelUsage, TraceRecorder
from steel.domain.enquiry import Enquiry, EscalationReason
from steel.domain.injection import remove_boundary_tags
from steel.domain.models import EnquiryStatus
from steel.domain.status import advance
from steel.llm.client import ConversationRequest, LLMError, ToolUse
from steel.storage import enquiries

PROMPT_ID = "agent"
PROMPT_VERSION = "v4"  # v2: draft_quote (Slice 5); v3: obvious injections at once; v4: corrections (F-012)
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_ID}_{PROMPT_VERSION}.md"
MAX_TURNS = 8


def run_agent(context: ToolContext, model: str) -> AgentRun:
    recorder = TraceRecorder(context.tenant_id)
    enquiry = context.load_enquiry()
    messages: list[dict[str, Any]] = [{"role": "user", "content": opening_message(context, enquiry)}]
    turns_used = 0
    while turns_used < MAX_TURNS and not context.decision_made():
        turns_used += 1
        try:
            response = context.llm.converse(_request(model, messages))
        except LLMError as error:
            _fail(context, f"agent_failed: {error}", recorder)
            break
        recorder.record(
            "llm_turn",
            response.model,
            result={"text": response.text(), "stop_reason": response.stop_reason},
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        tool_uses = response.tool_uses()
        if not tool_uses:
            break
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [_run_tool(context, use, recorder) for use in tool_uses]})

    if not context.decision_made():
        reason = (
            EscalationReason.AGENT_STEP_LIMIT if turns_used >= MAX_TURNS else EscalationReason.AGENT_GAVE_NO_DECISION
        )
        escalate_enquiry(context, reason, f"agent ended after {turns_used} turns without create_rfq or escalation")
        recorder.record("system", "escalate_to_human", arguments={"reason": reason})
    return AgentRun(
        enquiry_id=context.enquiry_id,
        tenant_id=context.tenant_id,
        model=model,
        prompt_version=PROMPT_VERSION,
        events=recorder.events,
        final_status=context.load_enquiry().status,
        rfq_id=context.rfq_id,
    )


def opening_message(context: ToolContext, enquiry: Enquiry) -> str:
    attachments = enquiries.list_attachments(context.connection, context.tenant_id, enquiry.id)
    attachment_note = ", ".join(item.filename for item in attachments) or "none"
    # Subject, body and file names are written by the customer, so none of them may close the fence (F-009).
    customer_text = remove_boundary_tags(f"Subject: {enquiry.subject}\n\n{enquiry.body}")
    return (
        f"Handle enquiry {enquiry.id}.\n"
        f"<enquiry>\n{customer_text}\n</enquiry>\n"
        f"Attachments (extract_specs reads them): {remove_boundary_tags(attachment_note)}"
    )


def _request(model: str, messages: list[dict[str, Any]]) -> ConversationRequest:
    return ConversationRequest(
        prompt_id=PROMPT_ID,
        prompt_version=PROMPT_VERSION,
        model=model,
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        messages=messages,
        tools=[tool.spec for tool in TOOLS.values()],
    )


def _run_tool(context: ToolContext, use: ToolUse, recorder: TraceRecorder) -> dict[str, Any]:
    """Run one tool call and return the tool_result block for the model. Refusals are errors, not crashes."""
    started = time.perf_counter()
    context.model_usage = None
    try:
        tool = TOOLS.get(use.name)
        if tool is None:
            raise ToolError(f"unknown tool {use.name!r}")
        result, is_error = tool.run(context, use.input), False
    except ToolError as refusal:
        result, is_error = {"error": str(refusal)}, True
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    recorder.record(
        "tool_call",
        use.name,
        arguments=use.input,
        result=result,
        is_error=is_error,
        latency_ms=latency_ms,
        **_model_usage_fields(context.model_usage),
    )
    # ensure_ascii=False: the model must read "Poznań", not "Poznań", or it copies the escape (F-010).
    content = json.dumps(result, ensure_ascii=False)
    return {"type": "tool_result", "tool_use_id": use.id, "content": content, "is_error": is_error}


def _model_usage_fields(usage: ModelUsage | None) -> dict[str, Any]:
    """Trace fields for a model the tool called itself; none for tools that call no model."""
    if usage is None:
        return {}
    return {
        "model": usage.model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "model_latency_ms": usage.latency_ms,
    }


def _fail(context: ToolContext, reason: str, recorder: TraceRecorder) -> None:
    enquiry = context.load_enquiry()
    if enquiry.status not in {EnquiryStatus.FAILED, EnquiryStatus.ESCALATED, EnquiryStatus.QUOTED}:
        history = advance(enquiry.status_history, EnquiryStatus.FAILED, context.clock())
        context.save_enquiry(enquiry.model_copy(update={"status_history": history, "failure_reason": reason}))
    recorder.record("system", "fail", result={"reason": reason}, is_error=True)
