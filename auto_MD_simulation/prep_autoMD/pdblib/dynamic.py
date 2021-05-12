#!/usr/bin/python
# from numpy import *
import numpy as np

SMALL_DIFF = 1e-9


def fill_dynamic_matrix(score_mat, gap_open=-1.5, gap_extend=-0.07, allow_begin_gap=True, allow_end_gap=True):
    # E: the score coming from up
    # F: the score coming from left
    # G: the score coming from diagonal
    # H: dynamic programming matrix

    l1, l2 = score_mat.shape

    # initialize matrices
    H = np.empty((l1 + 1, l2 + 1), dtype=float)
    E = np.empty((l1 + 1, l2 + 1), dtype=float)
    F = np.empty((l1 + 1, l2 + 1), dtype=float)
    G = np.empty((l1 + 1, l2 + 1), dtype=float)
    H[0, 0] = 0.0
    E[0, :] = -99999.0
    F[:, 0] = -99999.0
    G[0, :] = -99999.0
    G[:, 0] = -99999.0

    if allow_begin_gap:
        H[1:l1 + 2, 0] = 0.0
        H[0, 1:l2 + 2] = 0.0
    else:
        H[1:l1 + 2, 0] = gap_open + gap_extend * np.arange(1, l1 + 1)
        H[0, 1:l2 + 2] = gap_open + gap_extend * np.arange(1, l2 + 1)
    E[1:l1 + 2, 0] = H[1:l1 + 2, 0]
    F[0, 1:l2 + 2] = H[0, 1:l2 + 2]

    # fill the matrix
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            # update E matrix
            if allow_end_gap and j == l2:
                E[i, j] = max(E[i - 1, j], F[i - 1, j], G[i - 1, j])
            else:
                E[i, j] = max(E[i - 1, j] + gap_extend, F[i - 1, j] + gap_open + gap_extend,
                              G[i - 1, j] + gap_open + gap_extend)

            # update F matrix
            if (allow_end_gap and i == l1):
                F[i, j] = max(E[i, j - 1], F[i, j - 1], G[i, j - 1])
            else:
                F[i, j] = max(E[i, j - 1] + gap_open + gap_extend, F[i, j - 1] + gap_extend,
                              G[i, j - 1] + gap_open + gap_extend)

            # update G matrix 
            score = score_mat[i - 1, j - 1]
            G[i, j] = H[i - 1, j - 1] + score
            # update H matrix
            H[i, j] = max(E[i, j], F[i, j], G[i, j])

    return E, F, G, H


# backtracking
def backtrack_dynamic_matrix_single(E, F, G, H, gap_open=-1.5, gap_extend=-0.07):
    l1, l2 = map(lambda x: x - 1, H.shape)
    i = l1
    j = l2
    B = [[], []]
    SMALL_DIFF = 1e-9
    while True:
        if i == 0 and j == 0:
            B[0] = B[0][::-1]
            B[1] = B[1][::-1]
            break
        elif i == 0:
            while j > 0:
                B[0].append(-1)
                B[1].append(j - 1)
                j -= 1
        elif j == 0:
            while i > 0:
                B[0].append(i - 1)
                B[1].append(-1)
                i -= 1
        elif abs(H[i, j] - G[i, j]) <= SMALL_DIFF:
            B[0].append(i - 1)
            B[1].append(j - 1)
            i -= 1
            j -= 1
        elif abs(H[i, j] - E[i, j]) <= SMALL_DIFF:
            B[0].append(i - 1)
            B[1].append(-1)
            i -= 1
        elif abs(H[i, j] - F[i, j]) <= SMALL_DIFF:
            B[0].append(-1)
            B[1].append(j - 1)
            j -= 1

    return B, H[l1, l2]


def backtrack_dynamic_matrix(E, F, G, H, gap_open=-1.5, gap_extend=-0.07):
    ret = []
    B = [[], []]
    l1, l2 = map(lambda x: x - 1, H.shape)
    backtrack_recursive(l1, l2, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)
    return ret, H[l1, l2]


def backtrack_recursive(i, j, ret, B, E, F, G, H, gap_open=-1.5, gap_extend=-0.07):
    if i == 0 and j == 0:
        ret.append([B[0][-1::-1], B[1][-1::-1]])  # reverse and deepcopy
        return ret
    elif i == 0:
        while j > 0:
            B[0].append(-1)  # no match
            B[1].append(j - 1)
            j -= 1
        backtrack_recursive(i, j, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)
    elif j == 0:
        while i > 0:
            B[0].append(i - 1)
            B[1].append(-1)
            i -= 1
        backtrack_recursive(i, j, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)
    else:
        seqlen = len(B[0])
        if abs(H[i, j] - E[i, j]) <= SMALL_DIFF:
            B[0] = B[0][0:seqlen]
            B[1] = B[1][0:seqlen]
            h = H[i, j]
            i_init = i  # save i for other branch
            while abs(h - E[i, j]) <= SMALL_DIFF and i > 0:
                B[0].append(i - 1)
                B[1].append(-1)
                i -= 1
                h -= gap_extend
            backtrack_recursive(i, j, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)
            i = i_init  # replace i with the original value
        if abs(H[i, j] - F[i, j]) <= SMALL_DIFF:
            B[0] = B[0][0:seqlen]
            B[1] = B[1][0:seqlen]
            h = H[i, j]
            j_init = j
            while abs(h - F[i, j]) <= SMALL_DIFF and j > 0:
                B[0].append(-1)
                B[1].append(j - 1)
                j -= 1
                h -= gap_extend
            backtrack_recursive(i, j, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)
            j = j_init
        if abs(H[i, j] - G[i, j]) <= SMALL_DIFF:
            B[0] = B[0][0:seqlen]
            B[1] = B[1][0:seqlen]
            B[0].append(i - 1)
            B[1].append(j - 1)
            backtrack_recursive(i - 1, j - 1, ret, B, E, F, G, H, gap_open=gap_open, gap_extend=gap_extend)


def fill_pdb_align_score_matrix(seq1, seq2):
    match_score = 10.0  # exact match
    match_b_score = 0.1  # 'b' matched
    non_match_score = float('-inf')

    score_mat = np.empty((len(seq1), len(seq2)), dtype=float)
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == 'b' or seq2[j] == 'b':
                score_mat[i, j] = match_b_score
            elif seq1[i] == seq2[j]:
                score_mat[i, j] = match_score
            else:
                try:
                    s1 = Res31[seq1[i]]
                    s2 = Res31[seq2[j]]
                    if s1 == s2:
                        diff = False
                    else:
                        diff = True
                except:
                    diff = True
                if diff:
                    score_mat[i, j] = non_match_score
                else:
                    score_mat[i, j] = match_score
    return score_mat


def align_seq(seq1, seq2):
    score_mat = fill_pdb_align_score_matrix(seq1, seq2)
    gap_pen = -0.001  # gap penalty
    E, F, G, H = fill_dynamic_matrix(score_mat, gap_open=gap_pen, gap_extend=0.0)
    align = backtrack_dynamic_matrix_single(E, F, G, H, gap_open=gap_pen, gap_extend=0.0)[0]

    ind1 = 0
    ind2 = 0
    aligned1 = []
    aligned2 = []
    for i in range(len(align[0])):
        r1 = align[0][i]
        r2 = align[1][i]
        is_blank1 = True
        is_blank2 = True
        if r1 != -1 and seq1[r1] != 'b':
            p1 = seq1[ind1]
            ind1 += 1
            is_blank1 = False
        if r2 != -1 and seq2[r2] != 'b':
            p2 = seq2[ind2]
            ind2 += 1
            is_blank2 = False
        if (not is_blank1) and (not is_blank2):
            aligned1.append((seq1.index(p1), p1))
            aligned2.append((seq2.index(p2), p2))
        elif not is_blank1:
            aligned1.append((seq1.index(p1), p1))
            aligned2.append(None)
        elif not is_blank2:
            aligned1.append(None)
            aligned2.append((seq2.index(p2), p2))

    return aligned1, aligned2


if __name__ == '__main__':
    seq1 = ['ASP', 'CYS', 'LEU', 'LYS', 'GLU', 'LEU', 'GLN', 'SER', 'LYS', 'LYS', 'GLU']
    seq2 = ['GLU', 'LEU', 'CYS', 'GLU', 'LEU', 'GLN', 'SER', 'LYS', 'LYS', 'GLU']

    ali1, ali2 = align_seq(seq1, seq2)
    for i in range(len(ali1)):
        print(i, ali1[i], ali2[i])
