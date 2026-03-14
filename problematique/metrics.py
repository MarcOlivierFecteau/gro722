#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def edit_distance(x, y):
    """
    Levenshtein distance. Time complexity: O(m*n). Space complexity: O(m*n).

    Args:
        x, y (str): The symbolic chains to compare.

    Returns:
        out (int): Levenshtein distance.
    """

    m, n = len(x), len(y)
    table = np.zeros(
        (m + 1, n + 1)
    )  # table[i, j] = minimum edits to transform x[:i] to y[:j]

    # Base cases: converting to/from empty strings
    for i in range(m + 1):
        table[i, 0] = i
    for j in range(n + 1):
        table[0, j] = j

    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1]
            else:
                table[i, j] = 1 + min(
                    table[i - 1, j],  # deletion
                    table[i, j - 1],  # insertion
                    table[i - 1, j - 1],  # substitution
                )

    return table[m, n]


def confusion_matrix(true, pred, ignore=[], show=False):
    """
    Compute the confusion matrix.

    Args:
        true (list[str]): Target vector.
        pred (list[str]): Prediction vector.
        ignore (list[str]): Classes excluded from the confusion matrix.
    Returns:
        confusion_mat (np.NDArray): Confusion matrix.
    """
    n = len(true)
    classes = list(set(true))
    num_classes = len(classes)
    confusion_mat = np.zeros((num_classes, num_classes))
    for i in range(n):
        if true[i] in ignore:
            continue
        confusion_mat[classes.index(true[i]), classes.index(pred[i])] += 1
    row_sums = np.sum(confusion_mat, axis=1)
    confusion_mat = np.divide(
        confusion_mat,
        row_sums[:, np.newaxis],
        out=np.full_like(confusion_mat, np.nan),
        where=row_sums[:, np.newaxis] != 0,
    )

    if show:
        __plot_confusion_matrix(confusion_mat, classes)

    return confusion_mat


def __plot_confusion_matrix(confusion_mat: np.typing.NDArray, classes: list[str]):
    num_classes = confusion_mat.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot()
    img = ax.imshow(confusion_mat, cmap="viridis")
    plt.colorbar(img)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")
    ax.set_xticks(range(num_classes), classes, rotation=45)
    ax.set_yticks(range(num_classes), classes)
    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    # Levenshtein distance example from Wikipedia
    x = "kitten"
    y = "sitting"
    distance = edit_distance(x, y)
    assert distance == 3, (
        f"Expected Levenshtein distance of {x} and {y}: 3 (got {distance})."
    )

    target = ["cat", "dog", "cat", "dog", "cat", "dog"]
    prediction = ["cat", "cat", "cat", "dog", "dog", "dog"]
    conf_mat = confusion_matrix(target, prediction, show=True)
    conf_mat = confusion_matrix(target, prediction, ignore=["cat"], show=True)
