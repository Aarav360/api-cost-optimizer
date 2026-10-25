"""Fetch the trained router's embedding model, once (PLAN_v4.md §13; Amendment A11, items 3 and 4).

Run from the project root:  uv run python -m analysis.fetch_model

Downloads the five files the router reads, at the pinned commit of `Alibaba-NLP/gte-modernbert-base`, into
`routers.trained.SNAPSHOT` (under `.cache/`, not committed), and prints each file's sha256. The repository's pickle is
never fetched. The router itself never fetches: it loads this snapshot offline, after checking these hashes against its
settings file. A file already there is kept if it is complete, so a rerun after a download that stopped is cheap.
Equivalent by hand:

    for f in config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json; do
      curl -L --fail -o ".cache/gte-modernbert-base/<commit>/$f" \
        "https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/<commit>/$f"; done
"""

import hashlib
import shutil
import sys
import urllib.request

from routers.trained import MODEL_COMMIT, MODEL_FILES, MODEL_REPOSITORY, SNAPSHOT


def main() -> int:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        url = f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_COMMIT}/{name}"
        target = SNAPSHOT / name
        if not target.exists():
            partial = target.with_name(name + ".part")
            with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
                shutil.copyfileobj(response, out)
            partial.rename(target)  # only a complete download gets the file's name
        print(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
