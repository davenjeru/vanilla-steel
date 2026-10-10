"""Record live Claude answers for every golden extraction case, for replay in tests, CI and evals.

Usage (reads ANTHROPIC_API_KEY from .env; the key is never printed):
    uv run --env-file .env python -m scripts.record_fixtures

Re-run after changing the prompt, the model or a case's email: each of those
changes the request, so the old recording no longer matches and replay fails
loudly until a new one exists. Commit the files written to fixtures/llm/.
"""

from evals.golden_cases import DATASETS_DIR, load_dataset
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.replay import RecordingLLMClient
from steel.settings import Settings


def main() -> None:
    settings = Settings()
    llm = RecordingLLMClient(AnthropicClient.from_environment(), settings.llm_fixtures_dir)
    cases = load_dataset(DATASETS_DIR / "extraction")
    print(f"Recording {len(cases)} cases with {settings.extraction_model} into {settings.llm_fixtures_dir}")

    for case in cases:
        document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
        result = extract_spec(document, llm, settings.extraction_model)
        usage = result.llm
        print(f"  {case.name}: {usage.input_tokens}+{usage.output_tokens} tokens, {usage.latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
