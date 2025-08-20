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

    # Initialize the DP table
    if banded_width == -1:
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]  
            # Initialize the first row and column
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i * indel_penalty
        for j in range(1, n + 1):
            dp[0][j] = j * indel_penalty
    else:
        dp = [[float('inf')] * (banded_width * 2 + 1) for _ in range(m + 1)]
        for j in range(banded_width + 1):
            dp[0][j] = j * indel_penalty
        for i in range(1, banded_width + 1):
            dp[i][0] = i * indel_penalty




    #fill in the DP table with either full or banded alignment
    if banded_width == -1:
        #fill in the DP table with full alignment
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = dp[i - 1][j - 1] + (match_award if seq1[i - 1] == seq2[j - 1] else sub_penalty)
                delete = dp[i - 1][j] + indel_penalty
                insert = dp[i][j - 1] + indel_penalty
                dp[i][j] = min(match, insert, delete)
    else:
        #fill in the DP table with banded alignment
        for i in range(1, m + 1):
            if i < banded_width + 1:
                start = 1
                stop = i + banded_width + 1
            elif i > m + 1 - banded_width - 1:
                start = i - m + banded_width
                stop = banded_width * 2 + 1
            else:
                start = 0
                stop = banded_width * 2 + 1
            for j in range(start, stop):
                if i < banded_width + 1:
                    match = dp[i - 1][j - 1] + (match_award if seq1[i - 1] == seq2[max(i - banded_width + j -1, 0)] else sub_penalty)
                    delete = dp[i - 1][j] + indel_penalty
                    insert = dp[i][j - 1] + indel_penalty
                    dp[i][j] = min(match, insert, delete)
                elif i > m + 1 - banded_width - 1: 
                    match = dp[i - 1][j - 1] + (match_award if seq1[i - 1] == seq2[min(i - banded_width + j -1, 2*banded_width + 1)] else sub_penalty)
                    delete = dp[i - 1][j] + indel_penalty
                    insert = dp[i][j - 1] + indel_penalty
                    dp[i][j] = min(match, insert, delete)
                else:
                    match = dp[i - 1][j] + (match_award if seq1[i - 1] == seq2[i - banded_width + j -1 ] else sub_penalty)
                    if j + 1 >= 2 * banded_width + 1:
                        delete = float('inf')
                    else: 
                        delete = dp[i - 1][j + 1] + indel_penalty
                    if j == 0:
                        insert = float('inf')
                    else:
                        insert = dp[i][j - 1] + indel_penalty
                    dp[i][j] = min(match, insert, delete)



    # Backtracking to reconstruct the optimal alignment
    alignment_seq1, alignment_seq2 = [], []
    i, j = m, n

    while i > 0 or j > 0:
        current_cost = dp[i][j]
        
        if i > 0 and j > 0 and (seq1[i - 1] == seq2[j - 1] or current_cost == dp[i - 1][j - 1] + sub_penalty):
            alignment_seq1.append(seq1[i - 1])
            alignment_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and current_cost == dp[i][j - 1] + indel_penalty:
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

    alignment_seq1 = alignment_seq1 if alignment_seq1 else None
    alignment_seq2 = alignment_seq2 if alignment_seq2 else None

    alignment_cost = dp[m][n]

    return alignment_cost, ''.join(alignment_seq1), ''.join(alignment_seq2)



if __name__ == "__main__":
    output = align('GGGGTTTTAAAACCCCTTTT', 'TTTTAAAACCCCTTTTGGGG', banded_width=2, gap='-')
    print(output)