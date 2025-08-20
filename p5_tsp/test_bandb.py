import math

from byu_pytest_utils import max_score
from tsp_core import Timer, generate_network, score_tour
from math import inf

from tsp_solve import greedy_tour, dfs, branch_and_bound, branch_and_bound_smart



@max_score(5)
def test_branch_and_bound():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    timer = Timer(10)
    stats = branch_and_bound(graph, timer)
    print("Branch and Bound stats:", stats)  # Debugging line
    #assert_valid_tours(graph, stats)

    scores = {
        tuple(stat.tour): stat.score
        for stat in stats
    }
    print("Generated tours:", scores.keys())  # Debugging line
    assert scores[0, 3, 2, 1, 4] == 21
    assert len(scores) == 1

test_branch_and_bound()