from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from boldsimnet import cli


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        first = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        self.first_path = self.root / "A.npy"
        self.second_path = self.root / "B.npy"
        self.labels_path = self.root / "labels.txt"
        np.save(self.first_path, first)
        np.save(self.second_path, first.T)
        self.labels_path.write_text(
            "Visual\nControl\nDefault\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _arguments(self, *, first: Path | None = None) -> list[str]:
        return [
            "compare",
            str(self.first_path if first is None else first),
            str(self.second_path),
            "--labels",
            str(self.labels_path),
        ]

    def _run(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = cli.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_compare_prints_json_to_stdout(self) -> None:
        status, stdout, stderr = self._run(self._arguments())

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["score"], 1.0)
        self.assertTrue(payload["ordered"])
        self.assertEqual(payload["matrix_convention"], "row=source, column=target")

    def test_compare_writes_output_json(self) -> None:
        output = self.root / "result.json"
        status, stdout, stderr = self._run(
            [*self._arguments(), "--output", str(output)]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(list(self.root.glob(".result.json.*.tmp")), [])

    def test_blank_label_returns_exit_two(self) -> None:
        self.labels_path.write_text("Visual\n\nDefault\n", encoding="utf-8")
        status, stdout, stderr = self._run(self._arguments())

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("boldsimnet: error:", stderr)
        self.assertIn("blank functional label at line(s) 2", stderr)

    def test_npz_archive_is_rejected(self) -> None:
        archive = self.root / "graphs.npz"
        np.savez(archive, adjacency=np.load(self.first_path, allow_pickle=False))
        status, stdout, stderr = self._run(self._arguments(first=archive))

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("expected one .npy array, not an archive", stderr)

    def test_missing_output_parent_returns_exit_two(self) -> None:
        output = self.root / "missing" / "result.json"
        status, stdout, stderr = self._run(
            [*self._arguments(), "--output", str(output)]
        )

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("output directory does not exist", stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
