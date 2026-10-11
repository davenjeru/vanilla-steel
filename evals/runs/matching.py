"""Run the matching eval and upload the run to Confident AI.

Usage: uv run python -m evals.runs.matching [--verified-only]

No LLM is involved: the matcher is deterministic code. It runs against the
deterministic seed stock (not the database), the same stock the golden labels
were written for. The freight rate is logged as a hyperparameter, because it is
the setting most likely to reorder offers.
"""

import json

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.metrics.matching.best_offer_comes_first import BestOfferComesFirst
from evals.metrics.matching.best_offer_in_the_top_three import BestOfferInTheTopThree
from evals.metrics.matching.finds_every_suitable_offer import FindsEverySuitableOffer
from evals.metrics.matching.no_unsuitable_offers import NoUnsuitableOffers
from evals.metrics.matching.offers_in_the_right_order import OffersInTheRightOrder
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
from evals.runs.shared.run_labels import current_git_revision, run_identifier
from evals.runs.shared.run_modes import evaluate_each, exit_if_replay_failed, only_verified, parse_run_options
from evals.runs.shared.trace_tags import trace_tags
from steel.domain.geography import locate_delivery
from steel.domain.matching import FREIGHT_EUR_PER_TONNE_KM, find_matches
from steel.domain.models import MaterialSpec
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock

DATASET = "matching"
COMPONENT_VERSION = "matcher-v1"
STOCK = build_stock(DEFAULT_SEED)
SUPPLIERS_BY_ID = {supplier.id: supplier for supplier in SUPPLIERS}


@observe(name="matching", description="run the matching eval")
def match_and_trace(golden: Golden, case: GoldenCase) -> str:
    """Offers go to the trace as {"offers": [stock IDs in rank order]}, the shape the metrics read."""
    spec = MaterialSpec(**json.loads(golden.input))
    delivery = locate_delivery(spec.delivery_city, spec.delivery_country)
    outcome = find_matches(spec, STOCK, SUPPLIERS_BY_ID, delivery)
    offers = json.dumps({"offers": [match.stock_item_id for match in outcome.matches]})
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case, "replay"),
        input=golden.input,
        output=offers,
        expected_output=golden.expected_output,
        metadata={
            "delivered_eur_per_t": [str(match.delivered_eur_per_t) for match in outcome.matches],
            "delivery_precision": delivery.precision,
            "warnings": outcome.warnings,
        },
    )
    return offers


def main() -> None:
    options = parse_run_options("Run the matching eval.", modes=("replay",))
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = load_dataset(dataset_dir)
    if options.verified_only:
        cases = only_verified(cases)
    cases_by_name = {case.name: case for case in cases}
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases])

    result = evaluate_each(
        dataset.evals_iterator(
            metrics=[
                NoUnsuitableOffers(),
                BestOfferComesFirst(),
                BestOfferInTheTopThree(),
                OffersInTheRightOrder(),
                FindsEverySuitableOffer(),
            ],
            hyperparameters={
                "freight_eur_per_tonne_km": str(FREIGHT_EUR_PER_TONNE_KM),
                "stock_seed": DEFAULT_SEED,
            },
            identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
        ),
        lambda golden: match_and_trace(golden, cases_by_name[golden.name]),
    )
    exit_if_replay_failed(result, options.mode)


if __name__ == "__main__":
    main()
