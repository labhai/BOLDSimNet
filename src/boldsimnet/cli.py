"""Command-line interface for the BOLDSimNet reference implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from .core import compare


def _load_adjacency(path: Path) -> np.ndarray:
    """Load one NumPy adjacency array without permitting pickled objects."""
    try:
        value = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise ValueError(f"adjacency file does not exist: {path}") from exc
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load adjacency file {path}: {exc}") from exc

    if not isinstance(value, np.ndarray):
        close = getattr(value, "close", None)
        if close is not None:
            close()
        raise ValueError(f"expected one .npy array, not an archive: {path}")
    return value


def _load_labels(path: Path) -> tuple[str, ...]:
    """Read exactly one non-empty functional label per line."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"label file does not exist: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read label file {path}: {exc}") from exc

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"label file is empty: {path}")
    labels = tuple(line.strip() for line in lines)
    blank_lines = [index for index, label in enumerate(labels, start=1) if not label]
    if blank_lines:
        shown = ", ".join(str(index) for index in blank_lines[:5])
        suffix = "..." if len(blank_lines) > 5 else ""
        raise ValueError(f"blank functional label at line(s) {shown}{suffix}: {path}")
    return labels


def _write_json(path: Path, rendered: str) -> None:
    """Atomically write JSON without creating an implicitly missing directory."""
    if not path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {path.parent}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise ValueError(f"cannot write output file {path}: {exc}") from exc


def _run_compare(arguments: argparse.Namespace) -> int:
    first = _load_adjacency(arguments.first)
    second = _load_adjacency(arguments.second)
    labels = _load_labels(arguments.labels)
    result = compare(first, second, labels)
    rendered = json.dumps(
        result.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    if arguments.output is None:
        sys.stdout.write(rendered)
    else:
        _write_json(arguments.output, rendered)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boldsimnet",
        description="Compare directed weighted graphs with BOLDSimNet.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser(
        "compare",
        help="compare two .npy adjacency matrices",
    )
    compare_parser.add_argument("first", type=Path, metavar="A.npy")
    compare_parser.add_argument("second", type=Path, metavar="B.npy")
    compare_parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        metavar="labels.txt",
        help="UTF-8 file with one functional-network label per atlas node",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        metavar="result.json",
        help="write JSON atomically instead of printing it to stdout",
    )
    compare_parser.set_defaults(handler=_run_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; expected user/input errors return exit status 2."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, OSError) as exc:
        print(f"boldsimnet: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
