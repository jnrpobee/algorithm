def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    print("Starting branch_and_bound")  # Debugging line
    n = len(edges)
    stats = []
    best_tour = None
    best_score = float('inf')
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(n)
    stack = []

    def reduce_matrix(matrix):
        row_min = [min(row) if any(cell != float('inf') for cell in row) else 0 for row in matrix]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != float('inf'):
                    matrix[i][j] -= row_min[i]
        col_min = [min(matrix[i][j] for i in range(n) if matrix[i][j] != float('inf')) if any(matrix[i][j] != float('inf') for i in range(n)) else 0 for j in range(n)]
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != float('inf'):
                    matrix[i][j] -= col_min[j]
        reduced_cost = sum(row_min) + sum(col_min)
        print(f"Reduced matrix: {matrix}, reduced cost: {reduced_cost}")  # Debugging line
        return reduced_cost

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
                    stack.append((new_cost, tour + [next_city], new_matrix))
                else:
                    n_nodes_pruned += 1
                    cut_tree.cut(tour + [next_city])
                print(f"Branching: tour={tour + [next_city]}, cost={new_cost}, matrix={new_matrix}")  # Debugging line

    def normalize_tour(tour):
        min_index = tour.index(min(tour))
        return tour[min_index:] + tour[:min_index]

    initial_matrix = [row[:] for row in edges]
    initial_cost = reduce_matrix(initial_matrix)
    for start in range(n):
        if timer.time_out():
            break
        stack.append((initial_cost, [start], initial_matrix))

    while stack and not timer.time_out():
        cost, tour, matrix = stack.pop()
        print(f"Popped from stack: cost={cost}, tour={tour}, matrix={matrix}")  # Debugging line
        branch(tour, cost, matrix)
    if best_tour is not None and len(best_tour) == n:
        normalized_tour = normalize_tour(best_tour)
        stats.append(SolutionStats(
            tour=tuple(normalized_tour),  # Convert tour to tuple
            score=best_score,
            time=timer.time(),
            max_queue_size=len(stack),
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    print("Graph:", edges)  # Debugging line
    print("Timer:", timer)  # Debugging line
    print("Generated stats:", stats)  # Debugging line
    print("Generated tours:", [stat.tour for stat in stats])  # Debugging line
    return stats
 
