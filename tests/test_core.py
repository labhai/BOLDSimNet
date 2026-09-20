from __future__ import annotations

import json
import unittest

import numpy as np

from boldsimnet import (
    UndefinedCentralityError,
    compare,
    node_cost,
    outgoing_eigenvector_centrality,
    perron_diagnostics,
)
from boldsimnet.core import (
    _active_graph,
    _delete_node,
    _insert_node,
    _substitute_nodes,
)


def _cycle(weights: tuple[float, float, float]) -> np.ndarray:
    adjacency = np.zeros((6, 6), dtype=np.float64)
    adjacency[0, 1], adjacency[1, 2], adjacency[2, 0] = weights
    return adjacency


class CentralityTests(unittest.TestCase):
    def test_outgoing_right_eigenvector_and_fixed_atlas_order(self) -> None:
        result = outgoing_eigenvector_centrality(_cycle((1.0, 2.0, 4.0)))
        np.testing.assert_allclose(result.values[:3], [0.2, 0.4, 0.4], atol=1e-8)
        np.testing.assert_array_equal(result.values[3:], 0.0)
        self.assertEqual(result.status, "unique_perron_root")
        self.assertEqual(result.principal_eigenspace_dimension, 1)

    def test_centrality_is_scale_invariant(self) -> None:
        original = _cycle((1.0, 2.0, 4.0))
        tiny = original * 1e-12
        np.testing.assert_allclose(
            outgoing_eigenvector_centrality(original).values,
            outgoing_eigenvector_centrality(tiny).values,
            atol=1e-8,
        )

    def test_zero_perron_root_can_have_unique_right_eigenvector(self) -> None:
        dag = np.asarray([[0.0, 1.0], [0.0, 0.0]])
        diagnostics = perron_diagnostics(dag)
        self.assertEqual(diagnostics["spectral_radius"], 0.0)
        self.assertEqual(diagnostics["principal_eigenspace_dimension"], 1)
        np.testing.assert_allclose(
            outgoing_eigenvector_centrality(dag).values,
            [1.0, 0.0],
            atol=1e-7,
        )

    def test_isolated_nodes_do_not_inflate_perron_eigenspace(self) -> None:
        dag = np.zeros((100, 100), dtype=np.float64)
        dag[0, 1] = 1.0
        diagnostics = perron_diagnostics(dag)
        self.assertEqual(diagnostics["principal_eigenspace_dimension"], 1)
        centrality = outgoing_eigenvector_centrality(dag)
        np.testing.assert_allclose(centrality.values[:2], [1.0, 0.0], atol=1e-7)
        np.testing.assert_array_equal(centrality.values[2:], 0.0)

    def test_repeated_perron_eigenspace_fails_closed(self) -> None:
        adjacency = np.zeros((4, 4), dtype=np.float64)
        adjacency[0, 1] = adjacency[1, 0] = 1.0
        adjacency[2, 3] = adjacency[3, 2] = 1.0
        with self.assertRaisesRegex(UndefinedCentralityError, "not unique"):
            outgoing_eigenvector_centrality(adjacency)

    def test_empty_graph_metadata_is_consistent_and_values_are_read_only(self) -> None:
        empty = np.zeros((4, 4), dtype=np.float64)
        result = outgoing_eigenvector_centrality(empty)
        self.assertEqual(
            perron_diagnostics(empty)["principal_eigenspace_dimension"], 0
        )
        self.assertEqual(result.principal_eigenspace_dimension, 0)
        self.assertEqual(result.status, "empty_graph_zero")
        with self.assertRaises(ValueError):
            result.values[0] = 1.0


class PaperAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = ["Network"] * 6

    def test_substitution_prefers_same_label_then_lower_index_on_tie(self) -> None:
        first = np.zeros((6, 6), dtype=np.float64)
        second = np.zeros((6, 6), dtype=np.float64)
        first[0, 1] = first[1, 0] = 1.0
        second[2, 3] = second[3, 2] = 1.0
        labels = ["A", "B", "A", "B", "unused", "unused"]
        cost, alignment = _substitute_nodes(
            _active_graph(first), _active_graph(second), labels
        )
        self.assertAlmostEqual(cost, 0.0, places=12)
        self.assertEqual(alignment, ((0, 2), (1, 3)))

    def test_substitution_equal_cost_candidates_use_lower_atlas_index(self) -> None:
        first = np.zeros((6, 6), dtype=np.float64)
        second = np.zeros((6, 6), dtype=np.float64)
        first[0, 1] = first[1, 0] = 1.0
        second[2, 3] = second[3, 2] = 1.0
        _, alignment = _substitute_nodes(
            _active_graph(first), _active_graph(second), ["same"] * 6
        )
        self.assertEqual(alignment[0], (0, 2))

    def test_delete_prefers_a_functionally_represented_candidate(self) -> None:
        larger = np.zeros((5, 5), dtype=np.float64)
        smaller = np.zeros((5, 5), dtype=np.float64)
        larger[0, 1] = larger[1, 0] = 1.0
        larger[2, 0] = 4.0
        larger[3, 0] = 0.1
        smaller[0, 1] = smaller[1, 0] = 1.0
        labels = ["A", "B", "A", "C", "unused"]
        updated, _, selected = _delete_node(
            _active_graph(larger), _active_graph(smaller), labels
        )
        self.assertEqual(selected, 2)
        self.assertNotIn(2, updated.active_nodes)
        np.testing.assert_array_equal(updated.adjacency[2, :], 0.0)
        np.testing.assert_array_equal(updated.adjacency[:, 2], 0.0)

    def test_delete_fallback_uses_total_degree_then_lower_index(self) -> None:
        larger = np.zeros((6, 6), dtype=np.float64)
        smaller = np.zeros((6, 6), dtype=np.float64)
        smaller[0, 1] = 1.0
        larger[0, 1] = 1.0
        larger[2, 0] = 1.0
        larger[3, 0] = larger[3, 1] = 1.0
        labels = ["A", "A", "B", "C", "D", "E"]
        _, _, selected = _delete_node(
            _active_graph(larger), _active_graph(smaller), labels
        )
        self.assertEqual(selected, 2)

    def test_delete_equal_degree_candidates_use_lower_atlas_index(self) -> None:
        larger = np.zeros((6, 6), dtype=np.float64)
        smaller = np.zeros((6, 6), dtype=np.float64)
        smaller[0, 1] = 1.0
        larger[0, 1] = 1.0
        larger[2, 0] = 1.0
        larger[3, 0] = 1.0
        labels = ["A", "A", "B", "C", "D", "E"]
        _, _, selected = _delete_node(
            _active_graph(larger), _active_graph(smaller), labels
        )
        self.assertEqual(selected, 2)

    def test_insert_copies_incoming_and_outgoing_candidate_edges(self) -> None:
        smaller = np.zeros((4, 4), dtype=np.float64)
        larger = np.zeros((4, 4), dtype=np.float64)
        smaller[0, 1] = 2.0
        larger[0, 1] = 1.0
        larger[2, 0] = 0.2
        larger[1, 2] = 0.3
        updated, cost, selected = _insert_node(
            _active_graph(smaller), _active_graph(larger)
        )
        self.assertEqual(selected, 2)
        self.assertAlmostEqual(cost, 0.5 / (2.0 + 1e-8), places=12)
        self.assertAlmostEqual(updated.adjacency[2, 0], 0.2)
        self.assertAlmostEqual(updated.adjacency[1, 2], 0.3)
        self.assertIn(2, updated.active_nodes)

    def test_insert_equal_cost_candidates_use_lower_atlas_index(self) -> None:
        smaller = np.zeros((5, 5), dtype=np.float64)
        larger = np.zeros((5, 5), dtype=np.float64)
        smaller[0, 1] = 2.0
        larger[0, 1] = 1.0
        larger[2, 0] = 0.5
        larger[3, 0] = 0.5
        _, _, selected = _insert_node(
            _active_graph(smaller), _active_graph(larger)
        )
        self.assertEqual(selected, 2)

    def test_equal_insertion_and_deletion_cost_chooses_deletion(self) -> None:
        smaller = np.zeros((4, 4), dtype=np.float64)
        larger = np.zeros((4, 4), dtype=np.float64)
        smaller[0, 1] = 2.0
        larger[0, 1] = 1.0
        larger[2, 0] = 1.0
        result = node_cost(smaller, larger, ["A"] * 4)
        self.assertEqual(len(result.edits), 1)
        self.assertEqual(result.edits[0].operation, "delete")
        self.assertEqual(result.edits[0].node, 2)
        self.assertAlmostEqual(
            result.edits[0].accepted_cost,
            result.edits[0].alternative_cost,
            places=15,
        )

    def test_reversed_cycle_uses_defined_score_components(self) -> None:
        first = _cycle((1.0, 1.0, 1.0))
        second = first.T.copy()
        self.assertFalse(np.array_equal(first, second))
        result = compare(first, second, self.labels)
        self.assertAlmostEqual(result.node_cost, 0.0, places=12)
        self.assertAlmostEqual(result.centrality_distance, 0.0, places=7)
        self.assertAlmostEqual(result.score, 1.0, places=7)

    def test_empty_graph_cases_follow_declared_operational_rule(self) -> None:
        empty = np.zeros((6, 6), dtype=np.float64)
        cycle = np.zeros((6, 6), dtype=np.float64)
        cycle[0, 1] = cycle[1, 0] = 1.0
        both_empty = compare(empty, empty, self.labels)
        self.assertEqual(both_empty.node_cost, 0.0)
        self.assertEqual(both_empty.centrality_distance, 0.0)
        self.assertEqual(both_empty.score, 1.0)

        one_empty = compare(empty, cycle, self.labels)
        self.assertAlmostEqual(one_empty.node_cost, 0.5, places=7)
        self.assertAlmostEqual(one_empty.centrality_distance, 0.5, places=7)
        self.assertAlmostEqual(one_empty.score, 0.5, places=7)

    def test_tiny_positive_weight_remains_an_active_edge(self) -> None:
        tiny = np.zeros((3, 3), dtype=np.float64)
        tiny[0, 1] = 1e-300
        empty = np.zeros((3, 3), dtype=np.float64)
        result = node_cost(tiny, empty, ["A", "B", "C"])
        self.assertEqual(result.original_active_nodes_first, 2)
        self.assertEqual(result.original_active_nodes_second, 0)

    def test_pair_computation_does_not_mutate_inputs_and_is_json_safe(self) -> None:
        first = _cycle((1.0, 2.0, 4.0))
        second = _cycle((2.0, 1.0, 3.0))
        first_before, second_before = first.copy(), second.copy()
        result = compare(first, second, self.labels)
        np.testing.assert_array_equal(first, first_before)
        np.testing.assert_array_equal(second, second_before)
        json.dumps(result.to_dict(), allow_nan=False)
        self.assertGreater(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        self.labels = ["A", "B"]

    def test_rejects_non_square_and_different_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "square"):
            compare(np.zeros((2, 3)), np.zeros((2, 3)), self.labels)
        with self.assertRaisesRegex(ValueError, "different shapes"):
            compare(self.valid, np.zeros((3, 3)), self.labels)

    def test_rejects_negative_nonfinite_and_self_loop_weights(self) -> None:
        invalid_cases = []
        negative = self.valid.copy()
        negative[0, 1] = -1.0
        invalid_cases.append(negative)
        nonfinite = self.valid.copy()
        nonfinite[0, 1] = np.nan
        invalid_cases.append(nonfinite)
        self_loop = self.valid.copy()
        self_loop[0, 0] = 1.0
        invalid_cases.append(self_loop)
        for adjacency in invalid_cases:
            with self.subTest(adjacency=adjacency):
                with self.assertRaises(ValueError):
                    compare(adjacency, self.valid, self.labels)

    def test_rejects_label_length_mismatch_and_empty_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            compare(self.valid, self.valid, ["A"])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            compare(self.valid, self.valid, ["A", ""])

    def test_rejects_complex_empty_and_overflowing_adjacencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "real-valued"):
            compare(
                self.valid.astype(np.complex128),
                self.valid,
                self.labels,
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            compare(np.zeros((0, 0)), np.zeros((0, 0)), [])
        overflowing = np.full((3, 3), 1e308)
        np.fill_diagonal(overflowing, 0.0)
        with self.assertRaisesRegex(ValueError, "exceeds float64"):
            compare(overflowing, overflowing, ["A", "B", "C"])

    def test_rejects_non_string_or_bare_string_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence of strings"):
            compare(self.valid, self.valid, "AB")
        with self.assertRaisesRegex(ValueError, "must be a string"):
            compare(self.valid, self.valid, ["A", None])
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            compare(self.valid, self.valid, ["A", "   "])


if __name__ == "__main__":
    unittest.main()
