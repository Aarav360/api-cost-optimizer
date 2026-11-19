"""The rehearsal's Jev: a fake server, settings on its address and stored train passes for the drift check (PLAN_v4.md
Amendment A19, items 2 and 13; issue #77, #84).

Not a check: a module `analysis/check_final_run.py` and `analysis/check_final_jev.py` import. `jev_rehearsal(world)` starts
`analysis/fake_jev.py`'s server on a loopback port, builds the settings of `analysis/check_jev_client.py` on its address with
the made-up world's train items as the noise sample, G+'s synthetic weights from `analysis/check_jev_plus.py`, and a client
whose waits are injected (a key made up here, never a real one). It then stores the two train passes of each router into the
world's results folder through the collection module's own writer, as Stage 6 left them, so the drift check has a first pass
and a second to measure D6 from.

The fake answers as `jev-1.14.0`, the pin, so the drift check's model rule passes; a check sets `fake.model` or scripts a
reply to break it. Every call to the fake is the frozen call building's body, so a check compares what arrived with the
bodies built by hand.

Nothing here touches the real store, the real benchmark, `results/` or the real Jev.
"""

import contextlib
import itertools
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from analysis import collect_jev
from analysis.backends import JevStandIn
from analysis.benchmark import read_rows
from analysis.check_jev_client import KEY, synthetic_data
from analysis.check_jev_plus import synthetic_answers, synthetic_weights
from analysis.drift_rule import JEV_PIN
from analysis.fake_jev import FakeJev, Reply, choice_answer, noul_answer, response_body
from analysis.router_names import G_PLUS, NAMED_ROUTERS, G
from analysis.world import World
from routers import jev, jev_call

G_ROUTER, G_PLUS_ROUTER = NAMED_ROUTERS[G], NAMED_ROUTERS[G_PLUS]
STAMP = "2026-11-18T09:30:00+01:00"  # every stored stand-in record's `collected_at` (the real writer takes the clock)
DATE = "Wed, 18 Nov 2026 08:30:00 GMT"  # the fake's `Date` header, so a check can find it in the record
BUDGETS = {G_ROUTER: 20_000, G_PLUS_ROUTER: 20_000}  # large: no made-up prompt is cut unless a check lowers a budget

G_ANSWERS = {"success_haiku": 0.91, "success_sonnet": 0.62, "success_opus_low": 0.33}


def g_body(haiku: float, sonnet: float, opus_low: float, *, model: str = JEV_PIN, input_tokens: int = 300) -> dict:
    """A 200 body answering G's three questions with these Noul values."""
    answers = {name: noul_answer(value) for name, value in zip(G_ANSWERS, (haiku, sonnet, opus_low), strict=True)}
    return response_body(answers, model=model, input_tokens=input_tokens)


def plus_answers(**changes: float) -> dict[str, dict]:
    """Every G+ question's answer, from `check_jev_plus.py`'s hand-worked ones, with the Noul values named in `changes`
    replaced (for instance `success_haiku_a=0.5`)."""
    answers = synthetic_answers()
    options = answers.pop("request_type")
    noul = {name: noul_answer(changes.get(name, value)) for name, value in answers.items()}
    return noul | {"request_type": choice_answer(options)}


def plus_body(*, model: str = JEV_PIN, input_tokens: int = 300, **changes: float) -> dict:
    return response_body(plus_answers(**changes), model=model, input_tokens=input_tokens)


def replies(bodies: Sequence[object]) -> list[Reply]:
    return [Reply(200, body) for body in bodies]


@dataclass
class JevRehearsal:
    """The fake server and what a world needs to use it: the stand-in for `Backends.jev`, the settings and the weights."""

    fake: FakeJev
    stand_in: JevStandIn
    waits: list[float]  # what the client's injected sleep was asked for

    @property
    def loaded(self) -> collect_jev.Loaded:
        return self.stand_in.loaded

    def items(self, world: World) -> list[collect_jev.Item]:
        """The noise sample's items, as the drift step reads them: the world's train items."""
        return collect_jev.noise_items(self.loaded, world.benchmark)


def settings_for(url: str, noise_ids: Sequence[str], budgets: Mapping[str, int] = BUDGETS) -> collect_jev.Loaded:
    data = synthetic_data(url)
    data["budget_bytes"] = dict(budgets)
    data["noise_sample"] = {"seed": 2026, "item_ids": list(noise_ids)}
    return collect_jev.Loaded(jev_call.parse_settings(data), jev.parse_client_settings(data), data)


class FakeBoundClient(jev.JevClient):
    """A client that starts the fake server when it is entered and stops it, its threads joined, when it leaves, so the
    fake's threads are alive only while a step is sending (grading runs with no other thread, A19 item 2.5)."""

    def __init__(self, fake: FakeJev, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.fake = fake

    def __enter__(self) -> "FakeBoundClient":
        self.fake.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *error: object) -> None:
        try:
            super().__exit__(*error)  # type: ignore[arg-type]
        finally:
            self.fake.__exit__(*error)  # type: ignore[arg-type]


@contextlib.contextmanager
def jev_rehearsal(
    world: World,
    *,
    first: Mapping[str, Sequence[object]] | None = None,
    second: Mapping[str, Sequence[object]] | None = None,
    model: str = JEV_PIN,
    budgets: Mapping[str, int] = BUDGETS,
) -> Iterator[JevRehearsal]:
    """The fake server built (it runs only inside a client's `with`), the stand-in built and train's two passes stored for
    both routers in `world.results`.

    `first` and `second` map a router's section name to the response bodies (in the world's train items' order) that its
    pass 1 and pass 2 got; a router without one gets the fake's default answers, the same for every item. The fake's scripted
    replies are empty when this returns control, and its `model` is `model`."""
    train = [row["item_id"] for row in read_rows(world.benchmark, split="train")]
    answers = {**{name: noul_answer(value) for name, value in G_ANSWERS.items()}, **plus_answers()}
    fake = FakeJev(answers)
    folder = tempfile.TemporaryDirectory()
    train_results = Path(folder.name)  # Stage 6's made-up train records, apart from the world's results folder
    fake.model, fake.date, fake.join_threads = model, DATE, True
    try:
        loaded = settings_for(fake.url, train, budgets)
        waits: list[float] = []
        ticks = itertools.count()
        weights = jev.parse_weights(synthetic_weights(), loaded.call)

        def client() -> jev.JevClient:
            return FakeBoundClient(fake, loaded.client, {jev.KEY_VARIABLE: KEY}, sleep=waits.append,
                                   clock=lambda: next(ticks) * 0.25, jitter=lambda: 0.0)  # fmt: skip

        rehearsal = JevRehearsal(fake, JevStandIn(loaded, weights, client, train_results), waits)
        items = rehearsal.items(world)
        for pass_number, bodies in ((1, first or {}), (2, second or {})):
            for router in collect_jev.ROUTERS:
                fake.script = replies(bodies.get(router, []))
                with client() as sender:
                    collect_jev.collect(items, router=router, split="train", loaded=loaded, client=sender,
                                        results_dir=train_results, pass_number=pass_number, now=lambda: STAMP)  # fmt: skip
        fake.received.clear()  # what a check looks at is what the final evaluation sent
        fake.request_ids = 0
        yield rehearsal
    finally:
        fake.stop()
        folder.cleanup()
