from collections import defaultdict

def align(
        seq1: str,
        seq2: str,
        match_award=-3,
        indel_penalty=5,
        sub_penalty=1,
        banded_width=-1,
        gap='-'
) -> tuple[float, str | None, str | None]:
    """
        Align seq1 against seq2 using Needleman-Wunsch
        Put seq1 on left (j) and seq2 on top (i)
        => matrix[i][j]
        :param seq1: the first sequence to align; should be on the "left" of the matrix
        :param seq2: the second sequence to align; should be on the "top" of the matrix
        :param match_award: how many points to award a match
        :param indel_penalty: how many points to award a gap in either sequence
        :param sub_penalty: how many points to award a substitution
        :param banded_width: banded_width * 2 + 1 is the width of the banded alignment; -1 indicates full alignment
        :param gap: the character to use to represent gaps in the alignment strings
        :return: alignment cost, alignment 1, alignment 2
    """
    
    m, n = len(seq1), len(seq2)

    # Initialize the DP table with default value of float('inf')
    dp = defaultdict(lambda: float('inf'))

    # Initialize the first row and column
    dp[(0, 0)] = 0
    for i in range(1, m + 1):
        dp[(i, 0)] = i * indel_penalty
    for j in range(1, n + 1):
        dp[(0, j)] = j * indel_penalty

    # Fill in the DP table with either full or banded alignment
    if banded_width == -1:
        # Fill in the DP table with full alignment
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i - 1 >= 0 and j - 1 >= 0:
                    match = dp[(i - 1, j - 1)] + (match_award if seq1[i - 1] == seq2[j - 1] else sub_penalty)
                else:
                    match = float('inf')
                
                if i - 1 >= 0:
                    delete = dp[(i - 1, j)] + indel_penalty
                else:
                    delete = float('inf')
                
                if j - 1 >= 0:
                    insert = dp[(i, j - 1)] + indel_penalty
                else:
                    insert = float('inf')
                
                dp[(i, j)] = min(match, insert, delete)
    else:
        # Fill in the DP table with banded alignment
        for i in range(1, m + 1):
            for j in range(max(1, i - banded_width), min(n + 1, i + banded_width + 1)):
                if i - 1 >= 0 and j - 1 >= 0:
                    match = dp[(i - 1, j - 1)] + (match_award if seq1[i - 1] == seq2[j - 1] else sub_penalty)
                else:
                    match = float('inf')
                
                if i - 1 >= 0 and (i - 1 >= j - banded_width and i - 1 <= j + banded_width):
                    delete = dp[(i - 1, j)] + indel_penalty
                else:
                    delete = float('inf')
                
                if j - 1 >= 0 and (i >= j - banded_width and i <= j + banded_width):
                    insert = dp[(i, j - 1)] + indel_penalty
                else:
                    insert = float('inf')
                
                dp[(i, j)] = min(match, insert, delete)

    # Backtracking to reconstruct the optimal alignment
    alignment_seq1, alignment_seq2 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        current_cost = dp[(i, j)]
        
        if i > 0 and j > 0 and (seq1[i - 1] == seq2[j - 1] or current_cost == dp[(i - 1, j - 1)] + sub_penalty):
            alignment_seq1.append(seq1[i - 1])
            alignment_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and current_cost == dp[(i, j - 1)] + indel_penalty:
            alignment_seq1.append(gap)
            alignment_seq2.append(seq2[j - 1])
            j -= 1
        else:
            alignment_seq1.append(seq1[i - 1])
            alignment_seq2.append(gap)
            i -= 1

    # Reverse the alignments to get the correct order
    alignment_seq1.reverse()
    alignment_seq2.reverse()

    alignment_cost = dp[(m, n)]

    return alignment_cost, ''.join(alignment_seq1), ''.join(alignment_seq2)
