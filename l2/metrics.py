#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def edit_distance(a: list[str] | list[int], b: list[str] | list[int]):
    """Distance de Levenshtein."""
    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    m, n = len(a), len(b)
    table = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(m + 1):
        table[i, 0] = i
    for j in range(n + 1):
        table[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                table[i, j] = table[i - 1, j - 1]
            else:
                table[i, j] = 1 + min(
                    table[i - 1, j],  # deletion
                    table[i, j - 1],  # insertion
                    table[i - 1, j - 1],  # substibution
                )

    return table[m, n]
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------


if __name__ == "__main__":
    a = list("allo")
    b = list("apollo2")
    c = edit_distance(a, b)
    assert c == 3, (
        f'Expected Levenshtein distance of "{"".join(a)}" and "{"".join(b)}": 3 (got {c}).'
    )

    print(f'Distance d\'édition entre "{"".join(a)}" et "{"".join(b)}": {c}')
