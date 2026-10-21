"""Reading a numbered amendment item out of PLAN_v4.md, for the checks that compare a document's numbers with the
committed outputs (CODING_STANDARDS.md, "Numbers in documents"; Amendment A8, item 12; Amendment A9, items 17 and 18).
"""

import re

from harness.client import PROJECT_ROOT

PLAN = PROJECT_ROOT / "PLAN_v4.md"


def amendment_item(plan: str, amendment: str, number: int) -> str:
    """Item `number` of amendment `amendment` ("A8", "A9"), from its number to the next item's, or to the end of the
    amendment: the next amendment's heading, or the rule after the last one."""
    text = re.split(r"\n## |\n---\n", plan[plan.index(f"## Amendment {amendment}") :], maxsplit=1)[0]
    start = text.index(f"\n{number}. **") + 1
    end = text.find(f"\n{number + 1}. **", start)
    return text[start : None if end == -1 else end]
