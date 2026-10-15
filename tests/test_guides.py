"""Execute complete English guide examples in isolated temporary storage."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "records-and-indexes.md",
        "connection-lifecycle.md",
        "scanning-stores.md",
        "lmdb-3.md",
        "bounded-values-and-schemas.md",
        "combining-and-tracking-stores.md",
        "pytorch.md",
        "command-line-views.md",
    ],
)
def test_guide_examples(name):
    source = Path(__file__).parents[1] / "guides" / name
    text = source.read_text(encoding="utf-8")
    if "import torch" in text:
        pytest.importorskip("torch")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks):
        exec(compile(block, f"{name}:{index}", "exec"), {})
