import math
from collections.abc import Iterator
from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree

MATRIX = dict[tuple[int, int], float]

def _reduce_matrix(matrix: MATRIX) -> tuple[float, MATRIX]:
    matrix = matrix.copy()
    score = 0

    # rows
    row_min = {}
    for (r, c), w in matrix.items():
        if r not in row_min or row_min[r] > w:
            row_min[r] = w  

    score += sum(row_min.values())

    for (r, c), w in matrix.items():
        matrix[r, c] -= row_min[r]

    # columns
    col_min = {}
    for (r, c), w in matrix.items():
        if c not in col_min or col_min[c] > w:
            col_min[c] = w

    score += sum(col_min.values())

    for (r, c), w in matrix.items():
        matrix[r, c] -= col_min[c]

    return score, matrix

class ReducedPartialPath:
    score: float
    tour: list[int]

    def __init__(self, score, tour, matrix, n):
        self.score = score
        self.tour = tour
        self.matrix = matrix
        self.n = n

    @staticmethod
    def create_partial_path(edges: list[list[float]], start_at: int):
        matrix = {
            (i, j): w
            for i, row in enumerate(edges)
            for j, w in enumerate(row)
            if not math.isinf(w)
        }

        score, matrix = _reduce_matrix(matrix)

        return ReducedPartialPath(
            score,
            [start_at],
            matrix,
            len(edges)
        )

    def _expand(self, from_node, to_node: int) -> 'ReducedPartialPath':
        score = self.score
        score += self.matrix.get((from_node, to_node), math.inf)

        matrix = {
            (r, c): w
            for (r, c), w in self.matrix.items()
            if r != from_node and c != to_node
        }

        if (to_node, from_node) in matrix:
            del matrix[to_node, from_node]

        rscore, matrix = _reduce_matrix(matrix)
        score += rscore

        return ReducedPartialPath(
            score,
            self.tour + [to_node],
            matrix,
            self.n
        )

    def expand(self) -> Iterator['ReducedPartialPath']:
        visited = set(self.tour)
        for next_node in range(self.n):
            if next_node in visited:
                continue
            yield self._expand(self.tour[-1], next_node)

def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    best_tour = None
    best_score = float('inf')
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    for start in range(len(edges)):
        if timer.time_out():
            return stats
        
        unvisited = set(range(len(edges)))
        unvisited.remove(start)
        tour = [start]
        current_city = start
        cost = 0

        while unvisited:
            next_city = min(unvisited, key=lambda x: edges[current_city][x])
            if math.isinf(edges[current_city][next_city]):
                n_nodes_pruned += 1
                cut_tree.cut(tour + [next_city])
                break

            cost += edges[current_city][next_city]
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            n_nodes_expanded += 1

        if len(tour) == len(edges):
            if not math.isinf(edges[current_city][start]):
                cost += edges[current_city][start]
                if cost < best_score:
                    best_score = cost
                    best_tour = tour

    if best_tour is not None:
        stats.append(SolutionStats(
            tour=best_tour,
            score=best_score,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            tour=[],
            score=math.inf,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        )]
    return stats


def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    best_tour = None
    best_score = float('inf')
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    def dfs_recursive(tour, cost):
        nonlocal best_tour, best_score, n_nodes_expanded, n_nodes_pruned

        if timer.time_out():
            return
        
        current_city = tour[-1]

        if len(tour) == len(edges):
            if not math.isinf(edges[tour[-1]][tour[0]]):
                cost += edges[tour[-1]][tour[0]]
                if cost < best_score:
                    best_score = cost
                    best_tour = tour
            return

        for next_city in range(len(edges)):
            if next_city not in tour:
                if math.isinf(edges[current_city][next_city]):
                    n_nodes_pruned += 1
                    cut_tree.cut(tour + [next_city])
                    continue
                n_nodes_expanded += 1

                dfs_recursive(tour + [next_city], cost + edges[current_city][next_city])

    for start in range(len(edges)):
        if timer.time_out():
            break

        dfs_recursive([start], 0)

    if best_tour is not None:
        stats.append(SolutionStats(
            tour=best_tour,
            score=best_score,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            tour=[],
            score=math.inf,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        )]
    return stats
