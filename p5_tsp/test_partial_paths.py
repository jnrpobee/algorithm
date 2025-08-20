from partial_paths import _reduce_matrix, ReducedPartialPath
import math


def test_reduce():
    matrix = {
        (0, 1): 8,
        (0, 2): 12,
        (0, 3): 4,
        (1, 0): 3,
        (1, 2): 7,
        (1, 3): 1,
        (2, 0): 2,
        (2, 1): 6,
        (2, 3): 4,
        (3, 1): 3,
        (3, 2): 5,
    }

    exp_matrix = {
        (0, 1): 4,
        (0, 2): 6,
        (0, 3): 0,
        (1, 0): 2,
        (1, 2): 4,
        (1, 3): 0,
        (2, 0): 0,
        (2, 1): 4,
        (2, 3): 2,
        (3, 1): 0,
        (3, 2): 0,
    }

    score, rmatrix = _reduce_matrix(matrix)
    #score, rmatrix = reduced_matrix(matrix, set(range(4)), set(range(4)))
    assert score == 12
    assert rmatrix == exp_matrix


def test_expand_inf():
    matrix = {
        (0, 1): 4,
        (0, 2): 6,
        (0, 3): 0,
        (1, 0): 2,
        (1, 2): 4,
        (1, 3): 0,
        (2, 0): 0,
        (2, 1): 4,
        (2, 3): 2,
        (3, 0): math.inf,
        (3, 1): 0,
        (3, 2): 0,
    }

    exp_matrix = {
        (0, 2): 4,
        (0, 3): 0,
        (1, 0): 0,
        (1, 2): 0,
        (2, 0): 0,
        (2, 3): 2,
    }

    pp = ReducedPartialPath(
        score=0,
        tour=[3],
        matrix=matrix,
        n=4
        # set(range(4)),
        # set(range(4))
    )

    next_pp = pp._expand(from_node=3, to_node=1)

    print("Actual matrix:", next_pp.matrix)
    print("Expected matrix:", exp_matrix)
    print("Actual score:", next_pp.score)
    print("Expected score:", 4)

    assert next_pp.score == 4
    assert next_pp.matrix == exp_matrix

