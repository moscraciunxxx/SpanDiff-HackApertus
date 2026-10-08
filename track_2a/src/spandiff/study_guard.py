"""Stop a finished study from loading Apertus again."""

from __future__ import annotations

import sys


def refuse_graded_load() -> None:
    """The study results are already written. Do not load the model.

    ``--load-model`` is the only override. The graded prediction files stay
    where they are either way.
    """
    if "--load-model" in sys.argv:
        return
    raise SystemExit(
        "This study does not load the model. The graded development files stay as they are."
    )
