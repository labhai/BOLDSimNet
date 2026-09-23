from __future__ import annotations

import unittest

import numpy as np

from boldsimnet import UndefinedCentralityError, compare
from boldsimnet.core import (
    _active_graph,
    _delete_node,
    _insert_node,
    _substitute_nodes,
    node_cost,
    outgoing_eigenvector_centrality,
)


def _cycle(weights: tuple[float, float, float]) -> np.ndarray:
    adjacency = np.zeros((6, 6), dtype=np.float64)
    adjacency[0, 1], adjacency[1, 2], adjacency[2, 0] = weights
    return adjacency


class CoreTests(unittest.TestCase):
    def test_outgoing_eigenvector_centrality(self) -> None:
        result = outgoing_eigenvector_centrality(_cycle((1.0, 2.0, 4.0)))
        np.testing.assert_allclose(result.values[:3], [0.2, 0.4, 0.4], atol=1e-8)

        positive = np.asarray(
            [[0.0, 3.0, 3.0], [7.0, 0.0, 6.0], [1.0, 8.0, 0.0]]
        )
        original = outgoing_eigenvector_centrality(positive)
        scaled = outgoing_eigenvector_centrality(positive * 1e-12)
        self.assertEqual(original.principal_eigenspace_dimension, 1)
        np.testing.assert_allclose(original.values, scaled.values, atol=1e-8)

    def test_nonunique_centrality_raises(self) -> None:
        adjacency = np.zeros((4, 4), dtype=np.float64)
        adjacency[0, 1] = adjacency[1, 0] = 1.0
        adjacency[2, 3] = adjacency[3, 2] = 1.0
        with self.assertRaises(UndefinedCentralityError):
            outgoing_eigenvector_centrality(adjacency)

    def test_substitution_uses_functional_labels(self) -> None:
        first = np.zeros((4, 4), dtype=np.float64)
        second = np.zeros((4, 4), dtype=np.float64)
        first[0, 1] = first[1, 0] = 1.0
        second[2, 3] = second[3, 2] = 1.0
        cost, alignment = _substitute_nodes(
            _active_graph(first),
            _active_graph(second),
            ["A", "B", "A", "B"],
        )
        self.assertAlmostEqual(cost, 0.0)
        self.assertEqual(alignment, ((0, 2), (1, 3)))

    def test_delete_node(self) -> None:
        larger = np.zeros((4, 4), dtype=np.float64)
        smaller = np.zeros((4, 4), dtype=np.float64)
        larger[0, 1] = larger[1, 0] = 1.0
        larger[2, 0] = 4.0
        larger[3, 0] = 0.1
        smaller[0, 1] = smaller[1, 0] = 1.0
        updated, _, selected = _delete_node(
            _active_graph(larger),
            _active_graph(smaller),
            ["A", "B", "A", "C"],
        )
        self.assertEqual(selected, 2)
        self.assertNotIn(2, updated.active_nodes)

    def test_insert_node(self) -> None:
        smaller = np.zeros((3, 3), dtype=np.float64)
        larger = np.zeros((3, 3), dtype=np.float64)
        smaller[0, 1] = 2.0
        larger[0, 1] = 1.0
        larger[2, 0] = 0.2
        larger[1, 2] = 0.3
        updated, cost, selected = _insert_node(
            _active_graph(smaller), _active_graph(larger)
        )
        self.assertEqual(selected, 2)
        self.assertAlmostEqual(cost, 0.5 / (2.0 + 1e-8))
        self.assertAlmostEqual(updated.adjacency[2, 0], 0.2)
        self.assertAlmostEqual(updated.adjacency[1, 2], 0.3)

    def test_equal_edit_cost_chooses_deletion(self) -> None:
        smaller = np.zeros((4, 4), dtype=np.float64)
        larger = np.zeros((4, 4), dtype=np.float64)
        smaller[0, 1] = 2.0
        larger[0, 1] = 1.0
        larger[2, 0] = 1.0
        result = node_cost(smaller, larger, ["A"] * 4)
        self.assertEqual(result.edits[0].operation, "delete")

    def test_compare_returns_score_components(self) -> None:
        empty = np.zeros((4, 4), dtype=np.float64)
        graph = np.zeros((4, 4), dtype=np.float64)
        graph[0, 1] = graph[1, 0] = 1.0
        result = compare(empty, graph, ["A"] * 4)
        self.assertAlmostEqual(result.node_cost, 0.5)
        self.assertAlmostEqual(result.centrality_distance, 0.5)
        self.assertAlmostEqual(result.score, 0.5)
        self.assertEqual(
            set(result.to_dict()),
            {"score", "node_cost", "centrality_distance", "ordered"},
        )

    def test_invalid_adjacency_is_rejected(self) -> None:
        valid = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        invalid = valid.copy()
        invalid[0, 1] = -1.0
        with self.assertRaises(ValueError):
            compare(invalid, valid, ["A", "B"])
        with self.assertRaises(ValueError):
            compare(valid, valid, ["A"])


if __name__ == "__main__":
    unittest.main()
