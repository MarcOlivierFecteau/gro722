#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def edit_distance(x: list[str] | list[int], y: list[str] | list[int]) -> int:
    """
    Levenshtein distance. Time complexity: O(m*n). Space complexity: O(m*n).

    Args:
        x, y (list[str]): The symbolic chains to compare.

    Returns:
        out (int): Levenshtein distance.
    """

    m, n = len(x), len(y)
    table = np.zeros((m + 1, n + 1), dtype=np.int32)
    # NOTE: table[i, j] = minimum edits to transform x[:i] to y[:j]

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


def confusion_matrix(
    truth: list[str],
    prediction: list[str],
    ignore: list[str] | None = [],
    classes: list[str] | None = None,
    show: bool = False,
):
    """
    Compute the confusion matrix.

    Args:
        true (list[str]): Target vector.
        pred (list[str]): Prediction vector.
        ignore (list[str] | None): Classes excluded from the confusion matrix.
        classes (list[str] | None): Optional explicit class order.
        show (bool): If True, display the confusion matrix.

    Returns:
        out (NDArray): Row-normalized confusion matrix.
    """
    if ignore is None:
        ignore = []

    if classes is None:
        # Stable order: first appearance across truth + prediction
        classes = []
        seen = set()
        for label in list(truth) + list(prediction):
            if label not in seen and label not in ignore:
                seen.add(label)
                classes.append(label)
    else:
        classes = [c for c in classes if c not in ignore]
    classes = sorted(classes)

    class2index = {c: i for i, c in enumerate(classes)}
    confusion_mat = np.zeros((len(classes), len(classes)), dtype=np.float32)

    for t, p in zip(truth, prediction):
        if t in ignore or p in ignore:
            continue
        if t not in class2index or p not in class2index:
            continue
        confusion_mat[class2index[t], class2index[p]] += 1

    row_sums = confusion_mat.sum(axis=1, keepdims=True)
    confusion_mat = np.divide(
        confusion_mat,
        row_sums,
        out=np.full_like(confusion_mat, np.nan),
        where=row_sums != 0,
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
    if all(len(c) == 1 for c in classes):  # No need to rotate when only letters
        ax.set_xticks(range(num_classes), classes, rotation=0)
    else:
        ax.set_xticks(range(num_classes), classes, rotation=45)
    ax.set_yticks(range(num_classes), classes)
    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    # Levenshtein distance example from Wikipedia
    x = "kitten"
    y = "sitting"
    distance = edit_distance(x.split(""), y.split(""))  # type: ignore
    assert distance == 3, (
        f"Expected Levenshtein distance of {x} and {y}: 3 (got {distance})."
    )

    target = ["cat", "dog", "cat", "dog", "cat", "dog"]
    prediction = ["cat", "cat", "cat", "dog", "dog", "dog"]
    conf_mat = confusion_matrix(target, prediction, show=True)
    conf_mat = confusion_matrix(target, prediction, ignore=["cat"], show=True)
