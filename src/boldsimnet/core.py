"""BOLDSimNet graph-comparison implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPSILON = 1e-8


class UndefinedCentralityError(ValueError):
    """Raised when Eq. (3) does not define a unique centrality vector."""


def _validated_adjacency(adjacency: np.ndarray) -> np.ndarray:
    try:
        raw = np.asarray(adjacency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Adjacency cannot be converted to an array: {exc}") from exc
    if np.iscomplexobj(raw):
        raise ValueError("Adjacency must contain real-valued weights")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Adjacency must contain real numeric weights: {exc}") from exc
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Adjacency must be square, got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("Adjacency must contain at least one atlas node")
    if not np.isfinite(values).all():
        raise ValueError("Adjacency contains NaN or infinite values")
    if np.any(values < 0):
        raise ValueError("BOLDSimNet requires nonnegative edge weights")
    if np.any(np.diag(values) != 0):
        raise ValueError("Self-loops are not supported")
    with np.errstate(over="ignore", invalid="ignore"):
        total_weight = float(values.sum(dtype=np.float64))
    if not np.isfinite(total_weight):
        raise ValueError("Adjacency weight sum exceeds float64 range")
    return values.copy()


def _validated_labels(labels: Sequence[str], n_nodes: int) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)):
        raise ValueError("Functional labels must be a sequence of strings, not text")
    try:
        result = tuple(labels)
    except TypeError as exc:
        raise ValueError("Functional labels must be a sequence of strings") from exc
    if len(result) != n_nodes:
        raise ValueError(f"Expected {n_nodes} functional labels, got {len(result)}")
    if any(not isinstance(value, str) for value in result):
        raise ValueError("Every functional label must be a string")
    if any(not value.strip() for value in result):
        raise ValueError("Functional labels must be non-empty and non-whitespace")
    return result


@dataclass
class _ActiveGraph:
    adjacency: np.ndarray
    active_nodes: tuple[int, ...]

    def copy(self) -> "_ActiveGraph":
        return _ActiveGraph(self.adjacency.copy(), tuple(self.active_nodes))

    @property
    def node_set(self) -> set[int]:
        return set(self.active_nodes)

    def submatrix(self) -> np.ndarray:
        if not self.active_nodes:
            return np.empty((0, 0), dtype=np.float64)
        indices = np.asarray(self.active_nodes, dtype=np.int64)
        return self.adjacency[np.ix_(indices, indices)]

    def total_weight(self) -> float:
        return float(np.abs(self.submatrix()).sum())

    def incident_strength(self, node: int) -> float:
        """Normalized incoming-plus-outgoing strength used by Algorithms 2--4."""
        if node not in self.node_set:
            raise KeyError(f"Node {node} is not active")
        active = np.asarray(self.active_nodes, dtype=np.int64)
        incoming = np.abs(self.adjacency[active, node]).sum()
        outgoing = np.abs(self.adjacency[node, active]).sum()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            strength = float(
                (incoming + outgoing) / (self.total_weight() + EPSILON)
            )
        if not np.isfinite(strength):
            raise ValueError(
                "Normalized incident strength exceeds float64 range; "
                "rescale the adjacency weights"
            )
        return strength

    def total_degree(self, node: int) -> int:
        """Unweighted in-degree plus out-degree in the current active graph."""
        if node not in self.node_set:
            raise KeyError(f"Node {node} is not active")
        active = np.asarray(self.active_nodes, dtype=np.int64)
        incoming = np.count_nonzero(self.adjacency[active, node])
        outgoing = np.count_nonzero(self.adjacency[node, active])
        return int(incoming + outgoing)


def _active_graph(adjacency: np.ndarray) -> _ActiveGraph:
    values = _validated_adjacency(adjacency)
    active = np.flatnonzero(
        np.any(values != 0, axis=0) | np.any(values != 0, axis=1)
    )
    return _ActiveGraph(values, tuple(int(value) for value in active))


@dataclass(frozen=True)
class CentralityResult:
    """Eq. (3)--(4) result on the fixed atlas node order."""

    values: np.ndarray
    status: str
    spectral_radius: float
    principal_eigenspace_dimension: int

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def _spectral_diagnostics(adjacency: np.ndarray) -> tuple[float, int, np.ndarray]:
    """Return active-graph Perron diagnostics and a full-atlas vector."""
    active = np.flatnonzero(
        np.any(adjacency != 0, axis=0) | np.any(adjacency != 0, axis=1)
    )
    if active.size == 0:
        return 0.0, 0, np.zeros(adjacency.shape[0], dtype=np.float64)

    active_adjacency = adjacency[np.ix_(active, active)]
    edge_scale = float(np.max(np.abs(active_adjacency), initial=0.0))
    scaled = active_adjacency / edge_scale
    eigenvalues = np.linalg.eigvals(scaled)
    scaled_radius = float(np.max(np.abs(eigenvalues), initial=0.0))
    spectral_radius = scaled_radius * edge_scale

    # For a nonnegative matrix, the Perron eigenvalue equals the spectral
    # radius. SVD gives a deterministic null-space basis for A-rho*I.
    operator = scaled - scaled_radius * np.eye(active.size)
    _, singular_values, right_vectors = np.linalg.svd(operator)
    largest_singular = float(singular_values.max(initial=0.0))
    numerical_floor = float(singular_values[-1])
    tolerance = (
        numerical_floor
        + np.finfo(np.float64).eps * max(operator.shape) * largest_singular
    )
    multiplicity = int(np.count_nonzero(singular_values <= tolerance))

    vector = np.zeros(adjacency.shape[0], dtype=np.float64)
    vector[active] = right_vectors[-1, :]
    return spectral_radius, multiplicity, vector


def outgoing_eigenvector_centrality(adjacency: np.ndarray) -> CentralityResult:
    """Compute the L1-normalized principal right eigenvector (Eqs. 3--4).

    Isolated atlas parcels receive zero. A non-empty graph must have a
    one-dimensional principal eigenspace; otherwise the manuscript formula is
    numerically non-unique and :class:`UndefinedCentralityError` is raised.
    """
    values = _validated_adjacency(adjacency)
    edge_count = int(np.count_nonzero(values))
    radius, multiplicity, vector = _spectral_diagnostics(values)

    if edge_count == 0:
        return CentralityResult(
            np.zeros(values.shape[0], dtype=np.float64),
            "empty_graph_zero",
            0.0,
            0,
        )
    if multiplicity != 1:
        raise UndefinedCentralityError(
            "Principal-right-eigenvector eigenspace is not unique "
            f"(spectral_radius={radius:.12g}, "
            f"principal_eigenspace_dimension={multiplicity})"
        )

    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) <= EPSILON:
        raise UndefinedCentralityError("Principal eigenvector is numerically zero")
    if vector[pivot] < 0:
        vector = -vector
    if vector.sum() < 0:
        vector = -vector
    if float(vector.min(initial=0.0)) < -1e-7:
        raise UndefinedCentralityError("Principal right eigenvector is not nonnegative")
    vector = np.maximum(vector, 0.0)
    isolated = ~(
        np.any(values != 0, axis=0) | np.any(values != 0, axis=1)
    )
    vector[isolated] = 0.0
    norm = float(vector.sum())
    if norm <= EPSILON:
        raise UndefinedCentralityError("Principal right eigenvector has zero L1 norm")
    return CentralityResult(
        vector / (norm + EPSILON),
        "unique_perron_root",
        radius,
        multiplicity,
    )


def _centrality_distance(first: CentralityResult, second: CentralityResult) -> float:
    """Eq. (5): half the L1 distance between normalized EC vectors."""
    if first.values.shape != second.values.shape:
        raise ValueError("Centrality vectors have different shapes")
    return float(0.5 * np.abs(first.values - second.values).sum())


def _substitute_nodes(
    first: _ActiveGraph,
    second: _ActiveGraph,
    labels: Sequence[str],
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Algorithm 2 with ascending atlas-index traversal and tie-breaking."""
    functional = _validated_labels(labels, first.adjacency.shape[0])
    common = first.node_set & second.node_set
    nonmatching = sorted(first.node_set - common)
    matched = set(common)
    alignment: list[tuple[int, int]] = []
    cost = 0.0

    for node in nonmatching:
        available = sorted(second.node_set - matched)
        same_label = [
            candidate
            for candidate in available
            if functional[candidate] == functional[node]
        ]
        candidates = same_label if same_label else available
        if not candidates:
            raise RuntimeError("Substitution has no unmatched candidate")
        strength = first.incident_strength(node)
        selected = min(
            candidates,
            key=lambda candidate: (
                abs(strength - second.incident_strength(candidate)),
                candidate,
            ),
        )
        cost += abs(strength - second.incident_strength(selected))
        alignment.append((node, selected))
        matched.add(selected)
    return float(cost), tuple(alignment)


def _delete_node(
    larger: _ActiveGraph,
    smaller: _ActiveGraph,
    labels: Sequence[str],
) -> tuple[_ActiveGraph, float, int | None]:
    """Algorithm 3; fallback degree is unweighted in-degree plus out-degree."""
    functional = _validated_labels(labels, larger.adjacency.shape[0])
    candidates = sorted(larger.node_set - smaller.node_set)
    if not candidates:
        return larger.copy(), float("inf"), None

    represented_labels = {functional[node] for node in smaller.active_nodes}
    redundant = [node for node in candidates if functional[node] in represented_labels]
    if redundant:
        selected = min(
            redundant,
            key=lambda node: (larger.incident_strength(node), node),
        )
    else:
        selected = min(candidates, key=lambda node: (larger.total_degree(node), node))

    cost = larger.incident_strength(selected)
    updated = larger.copy()
    updated.active_nodes = tuple(node for node in updated.active_nodes if node != selected)
    updated.adjacency[selected, :] = 0.0
    updated.adjacency[:, selected] = 0.0
    return updated, float(cost), int(selected)


def _candidate_edges(
    node: int,
    smaller: _ActiveGraph,
    larger: _ActiveGraph,
) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    common_active = sorted(smaller.node_set & larger.node_set)
    for other in common_active:
        outgoing = float(larger.adjacency[node, other])
        incoming = float(larger.adjacency[other, node])
        if outgoing != 0:
            edges.append((node, other, outgoing))
        if incoming != 0:
            edges.append((other, node, incoming))
    return edges


def _insert_node(
    smaller: _ActiveGraph,
    larger: _ActiveGraph,
) -> tuple[_ActiveGraph, float, int | None]:
    """Algorithm 4 using the current pair-specific larger graph."""
    denominator = smaller.total_weight() + EPSILON
    choices: list[tuple[float, int, list[tuple[int, int, float]]]] = []
    for node in sorted(larger.node_set - smaller.node_set):
        edges = _candidate_edges(node, smaller, larger)
        if not edges:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cost = sum(abs(weight) for _, _, weight in edges) / denominator
        if not np.isfinite(cost):
            raise ValueError(
                "Normalized insertion cost exceeds float64 range; "
                "rescale the adjacency weights"
            )
        choices.append((float(cost), int(node), edges))
    if not choices:
        return smaller.copy(), float("inf"), None

    cost, selected, edges = min(choices, key=lambda item: (item[0], item[1]))
    updated = smaller.copy()
    updated.active_nodes = tuple(sorted((*updated.active_nodes, selected)))
    for source, target, weight in edges:
        updated.adjacency[source, target] = weight
    return updated, float(cost), int(selected)


@dataclass(frozen=True)
class EditOperation:
    operation: str
    graph_origin: str
    node: int | None
    accepted_cost: float
    alternative_cost: float | None


@dataclass(frozen=True)
class NodeCostResult:
    """Algorithms 1--4 and the normalized node cost."""

    node_cost: float
    substitution_cost: float
    equalization_cost: float
    original_active_nodes_first: int
    original_active_nodes_second: int
    final_active_nodes: int
    alignment_from_graph: str
    alignment_to_graph: str
    alignment: tuple[tuple[int, int], ...]
    edits: tuple[EditOperation, ...]


def node_cost(
    first: np.ndarray,
    second: np.ndarray,
    functional_labels: Sequence[str],
) -> NodeCostResult:
    """Compute Algorithms 1--4 and the manuscript's normalized node cost."""
    first_graph = _active_graph(first)
    second_graph = _active_graph(second)
    if first_graph.adjacency.shape != second_graph.adjacency.shape:
        raise ValueError("Adjacency matrices have different shapes")
    labels = _validated_labels(functional_labels, first_graph.adjacency.shape[0])
    n_first = len(first_graph.active_nodes)
    n_second = len(second_graph.active_nodes)
    edits: list[EditOperation] = []

    if n_first == n_second:
        substitution_cost, alignment = _substitute_nodes(
            first_graph, second_graph, labels
        )
        final_count = n_first
        alignment_from, alignment_to = "first", "second"
    else:
        if n_first < n_second:
            smaller, larger = first_graph.copy(), second_graph.copy()
            smaller_origin, larger_origin = "first", "second"
        else:
            smaller, larger = second_graph.copy(), first_graph.copy()
            smaller_origin, larger_origin = "second", "first"

        while len(smaller.active_nodes) != len(larger.active_nodes):
            deleted, deletion_cost, deleted_node = _delete_node(
                larger, smaller, labels
            )
            inserted, insertion_cost, inserted_node = _insert_node(smaller, larger)
            # The manuscript chooses deletion on an exact IC/DC tie.
            if deletion_cost > insertion_cost:
                smaller = inserted
                edits.append(
                    EditOperation(
                        "insert",
                        smaller_origin,
                        inserted_node,
                        float(insertion_cost),
                        float(deletion_cost),
                    )
                )
            else:
                larger = deleted
                edits.append(
                    EditOperation(
                        "delete",
                        larger_origin,
                        deleted_node,
                        float(deletion_cost),
                        float(insertion_cost) if np.isfinite(insertion_cost) else None,
                    )
                )
            if len(edits) > n_first + n_second:
                raise RuntimeError("Node equalization failed to converge")

        substitution_cost, alignment = _substitute_nodes(smaller, larger, labels)
        final_count = len(smaller.active_nodes)
        alignment_from, alignment_to = smaller_origin, larger_origin

    equalization_cost = float(sum(edit.accepted_cost for edit in edits))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (substitution_cost + equalization_cost) / (
            n_first + n_second + EPSILON
        )
    if not np.isfinite(equalization_cost) or not np.isfinite(normalized):
        raise ValueError(
            "Node cost exceeds float64 range; rescale the adjacency weights"
        )
    return NodeCostResult(
        float(normalized),
        float(substitution_cost),
        equalization_cost,
        n_first,
        n_second,
        final_count,
        alignment_from,
        alignment_to,
        alignment,
        tuple(edits),
    )


@dataclass(frozen=True)
class BOLDSimNetResult:
    """Components of the ordered BOLDSimNet score in Eq. (6)."""

    score: float
    node_cost: float
    centrality_distance: float
    node_cost_details: NodeCostResult
    centrality_first: CentralityResult
    centrality_second: CentralityResult

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "ordered": True,
            "score": self.score,
            "node_cost": self.node_cost,
            "centrality_distance": self.centrality_distance,
        }


def compare(
    first: np.ndarray,
    second: np.ndarray,
    functional_labels: Sequence[str],
) -> BOLDSimNetResult:
    """Compute the ordered BOLDSimNet score defined by Eq. (6)."""
    left = _validated_adjacency(first)
    right = _validated_adjacency(second)
    if left.shape != right.shape:
        raise ValueError("Adjacency matrices have different shapes")
    labels = _validated_labels(functional_labels, left.shape[0])

    node_result = node_cost(left, right, labels)
    centrality_left = outgoing_eigenvector_centrality(left)
    centrality_right = outgoing_eigenvector_centrality(right)
    distance = _centrality_distance(centrality_left, centrality_right)
    score = 1.0 / (1.0 + node_result.node_cost + distance)
    return BOLDSimNetResult(
        float(score),
        node_result.node_cost,
        distance,
        node_result,
        centrality_left,
        centrality_right,
    )
