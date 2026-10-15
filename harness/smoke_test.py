"""One Haiku call to check the API key, the request shape, and the response store.

Run from the project root:  uv run python -m harness.smoke_test

The first run makes one real call (a fraction of a cent). Re-runs read the
stored response and make no call, which is the cache check.
"""

import sys

from harness.client import PROJECT_ROOT, answer_text, build_request, call, cost_usd, load_models, make_client
from harness.store import ResponseStore

PROMPT = "What is 17 * 23? Reply with just the number."
EXPECTED = "391"


def main() -> int:
    models = load_models()
    config = models["configs"]["haiku"]
    store = ResponseStore(PROJECT_ROOT / "results" / "responses" / "smoke")
    client = make_client()

    request = build_request(config, PROMPT, max_tokens=2000)
    record, from_cache = call(client, store, request)
    response = record["response"]
    usage = response["usage"]
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")

    print(f"source         {'response store (no API call)' if from_cache else 'API call'}")
    print(f"stored at      {store.path_for(request, 0).relative_to(PROJECT_ROOT)}")
    print(f"request_id     {record['request_id']}")
    print(f"model          requested {request['model']}, reported {response['model']}")
    print(f"effort         {request['output_config']['effort']}")
    print(f"stop_reason    {response['stop_reason']}")
    print(f"answer         {answer_text(response)!r}")
    print(f"tokens         input {usage['input_tokens']}, output {usage['output_tokens']} (thinking {thinking})")
    print(f"latency        {record['latency_s']:.2f}s (from the original call)")
    print(f"cost           ${cost_usd(record, models['prices']):.6f}")

    problems = []
    if response["model"] != request["model"]:
        problems.append("reported model differs from the pinned ID")
    if response["stop_reason"] != "end_turn":
        problems.append(f"stop_reason is {response['stop_reason']}")
    if answer_text(response).strip() != EXPECTED:
        problems.append(f"answer is not {EXPECTED}")

    # Second call with the identical request must come from the store.
    again, again_from_cache = call(client, store, request)
    if not again_from_cache or again["request_id"] != record["request_id"]:
        problems.append("identical request was not served from the store")
    else:
        print("cache check    identical request served from the store, no second call")

    for problem in problems:
        print(f"PROBLEM        {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
