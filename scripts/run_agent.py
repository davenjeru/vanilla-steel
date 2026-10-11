"""Run one golden case's email through the agent and print what it did.

Usage:
    uv run python -m scripts.run_agent --case agent-001-s355-plate-rfq           # replay recordings
    uv run --env-file .env python -m scripts.run_agent --case ... --live          # call Claude, record

Works for extraction cases and agent cases. Agent cases also get their earlier
enquiries (for duplicate and data separation cases) and show the expected outcome.
--live calls Claude and saves every answer to fixtures/agent/<case>/, so the case replays later without a key.
"""

import argparse
import json

from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.extraction.extractor import EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient, clear_case_recordings
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings

DATASETS_WITH_EMAILS = ["agent", "extraction"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="agent or extraction golden case name")
    parser.add_argument("--live", action="store_true", help="call Claude and record the answers")
    arguments = parser.parse_args()

    settings = Settings()
    case = find_case(arguments.case)
    if case.email is None:
        raise SystemExit(f"{case.name} has no email")
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    result = handle_in_sandbox(
        document.subject,
        document.body,
        document.attachments,
        _llm(arguments.live, settings, case.name),
        settings,
        case.earlier_enquiries_for_sandbox(),
    )
    print_result(result)
    if case.labels.expected_agent:
        print_expectation(case)


def find_case(name: str) -> GoldenCase:
    for dataset in DATASETS_WITH_EMAILS:
        for case in load_dataset(DATASETS_DIR / dataset):
            if case.name == name:
                return case
    raise SystemExit(f"no case named {name} in {DATASETS_WITH_EMAILS}")


def _llm(live: bool, settings: Settings, case_name: str) -> LLMClient:
    """The agent conversation is recorded per case (finding F-007); a live run first clears the case's old one."""
    if live:
        clear_case_recordings(settings.llm_fixtures_dir, case_name)
        return RecordingLLMClient(AnthropicClient.from_environment(), settings.llm_fixtures_dir, case=case_name)
    return ReplayLLMClient(settings.llm_fixtures_dir, case=case_name)


def print_result(result: SandboxResult) -> None:
    enquiry = result.enquiry
    print(f"Status: {enquiry.status}", end="")
    if enquiry.escalation_reason:
        print(f" ({enquiry.escalation_reason}: {enquiry.escalation_detail})", end="")
    print(f"\nRFQ: {result.rfq.id + ' for ' + result.rfq.stock_item_id if result.rfq else 'none'}")
    input_tokens, output_tokens = result.run.total_tokens()
    print(f"Tokens: {input_tokens} in, {output_tokens} out\n\nTrace:")
    for event in result.run.events:
        detail = json.dumps(event.arguments if event.kind == "tool_call" else event.result, ensure_ascii=False)
        error = "  ERROR" if event.is_error else ""
        print(f"  {event.seq:>2} {event.kind:<9} {event.name:<18} {detail[:110]}{error}")


def print_expectation(case: GoldenCase) -> None:
    """What the golden case expects, for eyeballing. Scoring is the agent eval's job."""
    expected = case.labels.expected_agent
    if expected is None:
        return
    outcome = expected.rfq_stock_item_id if expected.decision == "rfq" else " or ".join(expected.escalation_reasons)
    print(f"\nExpected: {expected.decision} ({outcome}) via {' -> '.join(expected.tool_path)}")
    if expected.forbidden_tools:
        print(f"Must never call: {', '.join(expected.forbidden_tools)}")
    if case.labels.known_limitation:
        print(f"Known limitation: {case.labels.known_limitation}")


if __name__ == "__main__":
    main()
