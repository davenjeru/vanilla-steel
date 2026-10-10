from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.matching.matching_outcome import read_matching_outcome


class FindsEverySuitableOffer(BaseMetric):
    """Exactly the expected offers are made: none missing (a lost sale), none extra.

    The score is the overlap between offered and expected (shared / all), so
    finding 2 of 3 with nothing extra scores 0.67. A pass needs the exact set.
    Order does not matter here; "Offers in the right order" checks that.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Finds every suitable offer"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_matching_outcome(test_case)
        expected, offered = set(outcome.expected_offers), set(outcome.offered)
        missing = sorted(expected - offered)
        extra = sorted(offered - expected)
        everything = expected | offered
        self.score = round(len(expected & offered) / len(everything), 4) if everything else 1.0
        self.success = not missing and not extra
        self.reason = _reason(missing, extra, len(expected))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _reason(missing: list[str], extra: list[str], expected_count: int) -> str:
    if not missing and not extra:
        return (
            f"All {expected_count} suitable offers found, nothing extra"
            if expected_count
            else "No offer expected and none made"
        )
    problems = []
    if missing:
        problems.append(f"missed {', '.join(missing)}")
    if extra:
        problems.append(f"also offered {', '.join(extra)}")
    sentence = "; ".join(problems)
    return sentence[0].upper() + sentence[1:]
