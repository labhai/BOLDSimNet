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
    def test_compare_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adjacency = np.asarray(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            )
            first = root / "A.npy"
            second = root / "B.npy"
            labels = root / "labels.txt"
            np.save(first, adjacency)
            np.save(second, adjacency)
            labels.write_text("Visual\nControl\nDefault\n", encoding="utf-8")

            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = cli.main(
                    ["compare", str(first), str(second), "--labels", str(labels)]
                )

            self.assertEqual(status, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                set(payload),
                {"score", "node_cost", "centrality_distance", "ordered"},
            )
            self.assertEqual(payload["score"], 1.0)
            self.assertTrue(payload["ordered"])

            output = root / "result.json"
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = cli.main(
                    [
                        "compare",
                        str(first),
                        str(second),
                        "--labels",
                        str(labels),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
