"""User-visible text of the interactive and rich command-line views.

Every string the shared views print lives in :class:`Messages`; ``ENGLISH`` is the
default. Override single strings with ``dataclasses.replace(ENGLISH, deep_help=...)``.
Strings with ``{name}`` fields are formatted with ``str.format``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Messages:
    """Text catalog for :mod:`amber_lmdb.ui`."""

    # Command-line help.
    description: str = "Check the dataset root and the feature stores of {name}."
    doctor_help: str = "Check paths, the dataset and the feature stores."
    deep_help: str = "Also read the data (decode audio, validate metadata)."
    root_help: str = "Dataset root to use instead of the environment variable."
    resources_help: str = "Resources root to use instead of the environment variable."
    dataset_help: str = "Browse the dataset in an interactive explorer."
    paths_help: str = "Show how {dataset_env} and {resources_env} resolve."
    features_help: str = "Browse feature stores interactively or list them."
    name_filter_help: str = "Only stores whose name contains this text."
    list_help: str = "Print the table instead of opening the explorer."
    detail_help: str = "Print the metadata and the features of each store."
    output_dir_help: str = "Feature store directory to use instead of the environment variable."
    no_entries_help: str = "Do not open the stores to count entries."
    json_help: str = "Print JSON instead of formatted text."
    version_help: str = "Show the version and exit."
    help_help: str = "Show this help and exit."
    usage_heading: str = "Usage"
    commands_heading: str = "Commands"
    arguments_heading: str = "Arguments"
    options_heading: str = "Options"

    # Errors and hints.
    error_prefix: str = "error"
    hint_prefix: str = "hint"
    interrupted: str = "Interrupted."
    hint_resources: str = "Set {env} or pass --output-dir."
    hint_terminal: str = "Run it in a terminal; scripts can use `features --list` or `--json`."
    hint_root: str = "Set {env} or pass --root."

    # Command output.
    interactive_only: str = "The {command} command needs an interactive terminal."
    store_dir_unresolved: str = "Cannot locate the feature stores: {error}"
    store_dir_missing: str = "The directory does not exist; no features have been written yet."
    no_matching_store: str = "No store matches '{name}'."
    no_stores: str = "No features have been written yet."
    reading_store: str = "Reading {name}..."
    feature_stores_title: str = "Feature stores"
    unset: str = "(unset)"
    resources_unset_paths: str = (
        "feature_root and the feature stores are unavailable until this is set."
    )
    missing_suffix: str = " (missing)"
    missing_entries: str = "missing: {entries}"
    dataset_root_title: str = "Dataset root"
    resources_root_title: str = "Resources root"
    dataset_open_failed: str = "Cannot open the dataset: {error}"
    indexing_dataset: str = "Indexing dataset..."

    # Store views.
    current_unreadable: str = "CURRENT is unreadable"
    meta_unreadable: str = "Could not read the meta key.{reason}"
    features_unreadable: str = "(the feature list needs a readable meta)"
    source_missing: str = "{file} does not exist"
    features_title: str = "Features"
    usage_title: str = "Usage"

    # Doctor.
    resources_unset: str = "{env} is not set; features cannot be read or written."
    root_unresolved: str = "The dataset root could not be resolved."
    resources_root_unset: str = "The resources root is not set."
    store_open_failed: str = "cannot open the LMDB store"
    format_mismatch: str = "format mismatch"
    index_unavailable: str = "The dataset index is unavailable."
    failures_found: str = "{count} FAIL result(s) found"
    all_passed: str = "All checks passed"
    tally: str = "{ok} ok · {warn} warn · {skip} skip · {seconds:.1f}s"
    store_total: str = "{count} stores · {size}"
    result_title: str = "Result"
    paths_section: str = "Paths"
    dataset_section: str = "Dataset"
    stores_section: str = "Feature stores"
    store_checks_progress: str = "Feature store checks"

    # Interactive explorers.
    copied: str = "Copied: {text}"
    path_copied: str = "Copied the store path."
    confirm_delete: str = "Delete this?\n{path}"
    confirm_delete_final: str = "Delete it for good? This cannot be undone.\n{path}"
    delete_failed: str = "Deletion failed: {error}"
    deleted: str = "Deleted {name}."
    store_open_error: str = "Could not read the store."
    item_open_error: str = "Could not open the item."
    hint_move: str = "move"
    hint_select: str = "select"
    hint_back: str = "back"
    hint_quit: str = "quit"
    hint_copy_path: str = "copy path"
    hint_delete: str = "delete"
    filter_placeholder: str = "Filter (substring)"
    explorer_title: str = "{name} Dataset Explorer"
    store_explorer_title: str = "{name} Feature Store Explorer"


ENGLISH = Messages()

__all__ = ["ENGLISH", "Messages"]
