import heapq
def find_shortest_path_with_heap(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the heap-based algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """

    priority_queue = [(0, source)]
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    paths = {source: [source]}
    visited = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == target:
            return paths[current_node], current_distance

        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor in visited:
                continue

            new_distance = current_distance + weight

            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
                paths[neighbor] = paths[current_node] + [neighbor]

    return [], float('inf')
 




    


def find_shortest_path_with_array(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the array-based (linear lookup) algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """
    unvisited_nodes = [(0, source)]
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    paths = {source: [source]}
    visited = set()

    while unvisited_nodes:
        current_distance, current_node = min(unvisited_nodes, key=lambda x: x[0])
        unvisited_nodes.remove((current_distance, current_node))

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == target:
            return paths[current_node], current_distance

        for neighbor, weight in graph.get(current_node, {}).items():
            if neighbor in visited:
                continue

            new_distance = current_distance + weight

            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                unvisited_nodes.append((new_distance, neighbor))
                paths[neighbor] = paths[current_node] + [neighbor]

    return [], float('inf')
