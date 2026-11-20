"""What a drift check refuses with (PLAN_v4.md Amendment A19, items 2 and 13; issue #77, #83).

Each drift check in the final evaluation returns its report or raises a `DriftRefusal`, which stops the run before any call
on test. The message names the check and the dev numbers that broke its rule, never a test outcome, so the runner shows it
(`analysis/final_run.py`'s `MESSAGE_SHOWN`). The way on from a refusal is a new amendment (A11 item 17, A15 item 17, A17
item 26): nothing here chooses a substitute. `DriftRefusal` is a `StepRefusal` (`analysis/final_run.py`), which is what puts
its message in `MESSAGE_SHOWN`.
"""

from analysis.final_run import StepRefusal


class DriftRefusal(StepRefusal):
    """A frozen part has drifted past what its amendment allows. `check` names the drift check."""

    def __init__(self, check: str, why: str):
        super().__init__(f"the drift check {check!r} refused: {why}")
        self.check = check
