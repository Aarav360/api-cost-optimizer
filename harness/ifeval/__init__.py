"""Google's official IFEval instruction checkers, vendored (Apache 2.0, see LICENSE here).

Source: github.com/google-research/google-research, instruction_following_eval/,
commit e6890f85757dd84e27ca6df2dd30651dafad28e0 (2026-07-22). Upstream sha256:

  instructions.py           60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf
  instructions_registry.py  ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc
  instructions_util.py      a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b

Changes from upstream, nothing else:

1. Imports point at this package instead of `instruction_following_eval`.
2. `instructions.py` uses stdlib `logging` instead of `absl.logging` (same `.error` call).
3. `instructions_util._get_sentence_tokenizer` loads Punkt from nltk's `punkt_tab`
   data, because nltk >= 3.9 refuses the pickled model upstream loads. Same model.

The strict scoring loop from upstream's evaluation_lib.py lives in
harness/evaluators.py, which also seeds langdetect so language checks are repeatable.
"""
