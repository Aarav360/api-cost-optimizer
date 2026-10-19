"""AI2's IFBench instruction checkers, vendored (Apache 2.0, see LICENSE here).

Source: github.com/allenai/IFBench, ifbench/, commit
1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d (2026-09-09). Upstream sha256:

  instructions.py           a76a3b300b531b95200e53427c6b1fbb72b671cf6d358796797c0fa4b6c7c44f
  instructions_registry.py  5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9
  instructions_util.py      6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930
  classic_instructions.py   d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973

Changes from upstream, nothing else:

1. Imports point at this package instead of `ifbench`, and this file replaces
   upstream's __init__.py (which only re-exported the modules and located its data).
2. `instructions.py` no longer creates a `.nltk_data` folder inside the package or
   sets NLTK_DATA at import.
3. `instructions_util.py` no longer downloads nltk data at import (upstream also
   fetched the pickled "punkt" package). harness/evaluators.py fetches what the
   checkers use into the project's .cache/ on first use.
4. `instructions_util._get_sentence_tokenizer` loads Punkt from `punkt_tab`, because
   nltk >= 3.9 refuses the pickled model upstream loads. Same model.
5. `classic_instructions.py` uses stdlib `logging` instead of `absl.logging`.

`classic_instructions.py` is IFBench's copy of Google's IFEval checkers; with
docstrings removed it is code-identical to harness/ifeval/instructions.py apart
from imports. The IFBench test set uses none of them, only the 58 checkers in
`instructions.py`. The strict scoring loop (upstream evaluation_lib.py) is shared
with IFEval in harness/evaluators.py.
"""
