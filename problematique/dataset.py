#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class HandwrittenWords(Dataset):
    """Ensemble de données de mots écrits à la main."""

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, filename: str):
        super().__init__()
        # Lecture du texte
        self.pad_symbol = pad_symbol = "<pad>"
        self.start_symbol = start_symbol = "<sos>"
        self.stop_symbol = stop_symbol = "<eos>"

        self.data = dict()
        self.int2sym: dict[int, str] = dict()

        num_letters = len(HandwrittenWords.ALPHABET)
        self.int2sym[num_letters] = start_symbol
        self.int2sym[num_letters + 1] = stop_symbol
        self.int2sym[num_letters + 2] = pad_symbol
        for i, char in enumerate(HandwrittenWords.ALPHABET):
            self.int2sym[i] = char
        self.sym2int = {v: k for k, v in self.int2sym.items()}

        with open(filename, "rb") as fp:
            self.data = pickle.load(fp)
            # NOTE: self.data := list[ list[str, array[(x, y)]] ]
            # échantillon
            #   cible (y): str
            #   coords (x): array[(x, y)] (2, T)
        # "Relativisation" des coordonnées (vectorisation)
        for i, sample in enumerate(self.data):
            self.data[i][1] = np.diff(sample[1], 1, axis=-1)
            # self.data := list[ target (str), coords (NDArray) ]

        self.maxlength = dict()
        self.maxlength["target"] = max(len(sample[0]) for sample in self.data)
        self.maxlength["coords"] = max(sample[1].shape[1] for sample in self.data)

        # Extraction des symboles
        self.symbols = set()
        for sample in self.data:
            self.symbols = self.symbols.union(set(sample[0]))

        self.data = [
            [[self.sym2int[char] for char in sample[0]], sample[1]]
            for sample in self.data
        ]  # symbol-to-token pour cibles

        self.num_symbols = len(self.sym2int) + 3

        # Ajout du padding aux séquences
        self.maxlength["target"] += 1
        self.maxlength["coords"] += 1
        for i, sample in enumerate(self.data):
            target, coords = sample
            # Padding target sequence with special symbols
            self.data[i][0] = (
                target
                + [self.sym2int[stop_symbol]]
                + [self.sym2int[pad_symbol]]
                * (self.maxlength["target"] - len(target) - 1)
            )

            # Padding "relative" input sequence with zeros to stay at last coordinate
            pad_len = self.maxlength["coords"] - coords.shape[1]

            if pad_len > 0:
                padding = np.zeros((2, pad_len))
                self.data[i][1] = np.hstack((coords, padding))
            else:
                self.data[i][1] = coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][1]), torch.tensor(self.data[idx][0])

    def visualisation(self, idx):
        # Visualisation des échantillons
        plt.figure()
        title_no_special = [
            self.int2sym[i]
            for i in self.data[idx][0]
            if i
            not in [
                self.sym2int[self.start_symbol],
                self.sym2int[self.stop_symbol],
                self.sym2int[self.pad_symbol],
            ]
        ]
        plt.plot(self.data[idx][1][0], self.data[idx][1][1], label="Original")
        plt.legend()
        plt.title("".join(title_no_special))
        plt.show(block=True)


if __name__ == "__main__":
    # Code de test pour aider à compléter le dataset
    a = HandwrittenWords("problematique/data_trainval.p")
    for i in range(10):
        a.visualisation(np.random.randint(0, len(a)))
