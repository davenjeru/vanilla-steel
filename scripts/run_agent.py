"""Run one golden case's email through the agent and print what it did.

Usage:
    uv run python -m scripts.run_agent --case en-001-st37-sheet-poznan          # replay recordings
    uv run --env-file .env python -m scripts.run_agent --case ... --live         # call Claude, record

--live calls Claude (extraction and every agent turn) and saves the answers to
fixtures/llm/, so the same case replays later without a key.
"""

import argparse
import json

from evals.runs.shared.golden_cases import DATASETS_DIR, load_dataset
from steel.extraction.extractor import EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="extraction golden case name")
    parser.add_argument("--live", action="store_true", help="call Claude and record the answers")
    arguments = parser.parse_args()

    settings = Settings()
    [case] = [case for case in load_dataset(DATASETS_DIR / "extraction") if case.name == arguments.case]
    if case.email is None:
        raise SystemExit(f"{case.name} has no email")
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    result = handle_in_sandbox(
        document.subject, document.body, document.attachments, _llm(arguments.live, settings), settings
    )
    print_result(result)


def _llm(live: bool, settings: Settings) -> LLMClient:
    if live:
        return RecordingLLMClient(AnthropicClient.from_environment(), settings.llm_fixtures_dir)
    return ReplayLLMClient(settings.llm_fixtures_dir)


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


if __name__ == "__main__":
    main()
