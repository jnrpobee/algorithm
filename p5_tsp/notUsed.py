import math
import heapq
import random

from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree


def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]

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

#    return []


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    n = len(edges)
    stats = []
    best_tour = None
    best_score = float('inf')
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(n)

    # def reduce_matrix(matrix):
    #         n = len(matrix) # added
    #         row_min = []
    #         for row in matrix:
    #             min_value = min(row)
    #             if min_value == float('inf'):
    #                 row_min.append(0)
    #             else:
    #                 row_min.append(min_value)
    #         print(f"Row min: {row_min}")
    #         for r in range(n):
    #             for c in range(n):
    #                 if matrix[r][c] != float('inf'):
    #                     matrix[r][c] -= row_min[r]
    #         print(f"Matrix after row reduction: {matrix}")
            
    #         col_min = []
    #         for c in range(n):
    #             col_values = [matrix[r][c] for r in range(n)]
    #             min_value = min(col_values)
    #             if min_value == float('inf'):
    #                 col_min.append(0)
    #             else:
    #                 col_min.append(min_value)
    #         print(f"Col min: {col_min}")
    #         for r in range(n):
    #             for c in range(n):
    #                 if matrix[r][c] != float('inf'):
    #                     matrix[r][c] -= col_min[c]
    #         print(f"Matrix after col reduction: {matrix}")
            
    def reduce_matrix(matrix):
            #n = len(matrix) # added
        row_min = [min(row) for row in matrix]
        #print(f"Row min: {row_min}")
        for r in range(n):
            for c in range(n):
                if matrix[r][c] != float('inf'):
                    matrix[r][c] -= row_min[r]
        #print(f"Matrix after row reduction: {matrix}")        
        col_min = [min(matrix[r][c] for r in range(n)) for c in range(n)]
        #print(f"Col min: {col_min}")
        for r in range(n):
            for c in range(n):
                if matrix[r][c] != float('inf'):
                    matrix[r][c] -= col_min[c]
        #print(f"Matrix after col reduction: {matrix}")
        return sum(row_min) + sum(col_min)
           

    def branch(tour, cost, matrix):
        nonlocal best_tour, best_score, n_nodes_expanded, n_nodes_pruned

        if timer.time_out():
            #print("Timer timed out")
            return
        
        if len(tour) == n:
            if not math.isinf(matrix[tour[-1]][tour[0]]):
                cost += matrix[tour[-1]][tour[0]]
                if cost < best_score:
                    best_score = cost
                    best_tour = tour
            return
        
        for next_city in range(n):
            if next_city not in tour:
                new_matrix = [row[:] for row in matrix]
                for r in range(n):
                    new_matrix[tour[-1]][r] = float('inf')
                    new_matrix[r][next_city] = float('inf')
                new_cost = cost + matrix[tour[-1]][next_city] + reduce_matrix(new_matrix)
                #if new_cost < best_score:
                #reduced_cost = reduce_matrix(new_matrix)
                #new_cost += reduced_cost
                #print(f"Tour: {tour + [next_city]}, Cost: {new_cost}, Reduced Cost: {reduced_cost}, Matrix: {new_matrix}")
                if new_cost < best_score:
                    n_nodes_expanded += 1
                    #print(f"Expanding node: {tour + [next_city]}, cost: {new_cost}")
                    branch(tour + [next_city], new_cost, new_matrix)
                else:
                    n_nodes_pruned += 1
                    #print(f"Pruning node: {tour + [next_city]}, cost: {new_cost}")
                    cut_tree.cut(tour + [next_city])

    initial_matrix = [row[:] for row in edges]
    initial_cost = reduce_matrix(initial_matrix)
    #print(f"Initial reduced cost: {initial_cost}")
    # for start in range(n):
    #     if timer.time_out():
    #         break
    branch([0], initial_cost, initial_matrix)

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
    

def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    n = len(edges)
    stats = []
    best_tour = None
    best_score = float('inf')
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(n)
    pq = []

    def reduce_matrix(matrix):
        row_min = [min(row) for row in matrix]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != float('inf'):
                    matrix[i][j] -= row_min[i]
        col_min = [min(matrix[i][j] for i in range(n)) for j in range(n)]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != float('inf'):
                    matrix[i][j] -= col_min[j]
        return sum(row_min) + sum(col_min)

    def branch(tour, cost, matrix):
        nonlocal best_tour, best_score, n_nodes_expanded, n_nodes_pruned

        if timer.time_out():
            return
        
        if len(tour) == n:
            if not math.isinf(matrix[tour[-1]][tour[0]]):
                cost += matrix[tour[-1]][tour[0]]
                if cost < best_score:
                    best_score = cost
                    best_tour = tour
            return
        
        for next_city in range(n):
            if next_city not in tour:
                new_matrix = [row[:] for row in matrix]
                for i in range(n):
                    new_matrix[tour[-1]][i] = float('inf')
                    new_matrix[i][next_city] = float('inf')
                new_cost = cost + matrix[tour[-1]][next_city] + reduce_matrix(new_matrix)
                if new_cost < best_score:
                    n_nodes_expanded += 1
                    heapq.heappush(pq, (new_cost, tour + [next_city], new_matrix))
                else:
                    n_nodes_pruned += 1
                    cut_tree.cut(tour + [next_city])

    initial_matrix = [row[:] for row in edges]
    initial_cost = reduce_matrix(initial_matrix)
    for start in range(n):
        if timer.time_out():
            break
        heapq.heappush(pq, (initial_cost, [start], initial_matrix))

    while pq and not timer.time_out():
        cost, tour, matrix = heapq.heappop(pq)
        branch(tour, cost, matrix)

    if best_tour is not None:
        stats.append(SolutionStats(
            tour=best_tour,
            score=best_score,
            time=timer.time(),
            max_queue_size=len(pq),
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
            max_queue_size=len(pq),
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        )]
    
    return stats



