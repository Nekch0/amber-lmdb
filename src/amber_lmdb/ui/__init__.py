"""Human-oriented views of stores: rich output, doctor reports and textual explorers.

Packages built on amber-lmdb customize these views instead of copying them:
:class:`StoreProfile` changes how stores are summarized, :class:`Messages` replaces the
text, :class:`Listing` and :class:`Detail` describe a browsable dataset, and
:class:`DatasetCLI` assembles the ``doctor``, ``dataset``, ``paths`` and ``features``
commands of a dataset package. The explorers live in
:mod:`amber_lmdb.ui.tui`.

Names load their module on first use, so a command line that only parses arguments
never imports Rich or Textual.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._common import ACCENT, Status, TreeNode, cap_list
    from .browse import Detail, Listing
    from .dataset_cli import DatasetCLI
    from .doctor import Check, Section, check_store_layout, check_store_meta, render_report
    from .messages import ENGLISH, Messages
    from .render import key_value_grid, print_section
    from .stores import (
        StoreProfile,
        StoreSummary,
        SummaryRow,
        default_summary,
        default_usage,
        describe_store,
        environment_rows,
        feature_nodes,
        feature_usage_comment,
        find_stores,
        meta_rows,
        meta_value,
    )
    from .views import store_grid

_MODULES = {
    "ACCENT": "_common",
    "Status": "_common",
    "TreeNode": "_common",
    "cap_list": "_common",
    "Detail": "browse",
    "Listing": "browse",
    "DatasetCLI": "dataset_cli",
    "Check": "doctor",
    "Section": "doctor",
    "check_store_layout": "doctor",
    "check_store_meta": "doctor",
    "render_report": "doctor",
    "ENGLISH": "messages",
    "Messages": "messages",
    "key_value_grid": "render",
    "print_section": "render",
    "StoreProfile": "stores",
    "StoreSummary": "stores",
    "SummaryRow": "stores",
    "default_summary": "stores",
    "default_usage": "stores",
    "describe_store": "stores",
    "environment_rows": "stores",
    "feature_nodes": "stores",
    "feature_usage_comment": "stores",
    "find_stores": "stores",
    "meta_rows": "stores",
    "meta_value": "stores",
    "store_grid": "views",
}


def __getattr__(name: str) -> Any:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_MODULES])


__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "ACCENT",
    "ENGLISH",
    "Check",
    "DatasetCLI",
    "Detail",
    "Listing",
    "Messages",
    "Section",
    "Status",
    "StoreProfile",
    "StoreSummary",
    "SummaryRow",
    "TreeNode",
    "cap_list",
    "check_store_layout",
    "check_store_meta",
    "default_summary",
    "default_usage",
    "describe_store",
    "environment_rows",
    "feature_nodes",
    "feature_usage_comment",
    "find_stores",
    "key_value_grid",
    "meta_rows",
    "meta_value",
    "print_section",
    "render_report",
    "store_grid",
]
