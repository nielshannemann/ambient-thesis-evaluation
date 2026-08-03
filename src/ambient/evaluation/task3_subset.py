"""Create deterministic Task-3 subsets from existing generation artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ambient.utils import write_json_atomic


def item_id(item: dict, index: int) -> str:
    value = item.get("id")
    if value is None:
        value = item.get("row_id")
    return str(index if value is None else value)


def read_ids(path: Path) -> set[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return {str(value) for value in json.loads(raw)}
    return {
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def run(args) -> int:
    with args.results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No Task-3 results found in {args.results_path}")

    indexed = [(item_id(item, index), item) for index, item in enumerate(results)]
    exclude_id_file = getattr(args, "exclude_id_file", None)
    excluded_ids: set[str] = set()
    if exclude_id_file is not None:
        excluded_ids = read_ids(exclude_id_file)
        source_ids = {identifier for identifier, _item in indexed}
        missing_exclusions = excluded_ids - source_ids
        if missing_exclusions:
            raise ValueError(
                f"Exclusion file contains {len(missing_exclusions)} IDs absent from "
                "the Task-3 artifact."
            )
        indexed = [
            (identifier, item)
            for identifier, item in indexed
            if identifier not in excluded_ids
        ]

    if args.id_file is not None:
        selected_ids = read_ids(args.id_file)
        overlap = selected_ids & excluded_ids
        if overlap:
            raise ValueError(
                f"ID and exclusion files overlap on {len(overlap)} Task-3 IDs."
            )
        selected = [(identifier, item) for identifier, item in indexed if identifier in selected_ids]
        found = {identifier for identifier, _item in selected}
        missing = selected_ids - found
        if missing:
            raise ValueError(f"ID file contains {len(missing)} IDs absent from the Task-3 artifact.")
    else:
        if args.sample_size is None:
            raise ValueError("Provide either --id-file or --sample-size.")
        if args.sample_size < 1 or args.sample_size > len(indexed):
            raise ValueError(f"sample_size must be in [1, {len(indexed)}]")
        rng = random.Random(args.selection_seed)
        selected_indices = set(rng.sample(range(len(indexed)), args.sample_size))
        selected = [row for index, row in enumerate(indexed) if index in selected_indices]

    metadata = dict(payload.get("metadata") or {})
    metadata["subset"] = {
        "source_results_path": str(args.results_path),
        "num_source_items": len(results),
        "num_selected_items": len(selected),
        "id_file": str(args.id_file) if args.id_file else None,
        "exclude_id_file": str(exclude_id_file) if exclude_id_file else None,
        "num_excluded_items": len(excluded_ids),
        "selection_seed": args.selection_seed if args.id_file is None else None,
    }
    output = {"metadata": metadata, "results": [item for _identifier, item in selected]}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, output)

    id_output = args.id_output or args.output_path.with_suffix(".ids.txt")
    id_output.parent.mkdir(parents=True, exist_ok=True)
    id_output.write_text("\n".join(identifier for identifier, _item in selected) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote {len(selected)} Task-3 items to {args.output_path}")
    print(f"[INFO] Wrote shared ID list to {id_output}")
    return 0
