#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re

import numpy as np
import torch
from torch.utils.data import Dataset


class Fr_En(Dataset):
    """Ensemble de données de mots/phrases en français et anglais."""

    def __init__(self, filename="l2/fra.txt", n_samp=2000, start=0, samplelen=[7, 10]):
        # Initialisation des variables
        self.pad_symbol = pad_symbol = "<pad>"
        self.start_symbol = start_symbol = "<sos>"
        self.stop_symbol = stop_symbol = "<eos>"

        symb_to_remove = set(["", " ", "\u202f"])
        data = dict()
        data["fr"] = {}
        data["en"] = {}
        data_cpt = 0
        df = {}

        # Lecture du texte
        with open(filename, encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                df[i] = line

        # Dictionnaires de symboles vers entiers (Tokenization)
        self.symb2int = {}
        self.symb2int["fr"] = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}
        self.symb2int["en"] = {start_symbol: 0, stop_symbol: 1, pad_symbol: 2}
        cpt_symb_fr = 3
        cpt_symb_en = 3

        for i in range(len(df)):
            en, fr, _ = df[i].split("\t")

            # Francais
            line = fr.lower()
            line = re.split(r"(\W)", line)
            line = list(filter(lambda x: x not in symb_to_remove, line))
            if len(line) < samplelen[0] or len(line) > samplelen[1]:
                continue
            for symb in line:
                if symb not in self.symb2int["fr"]:
                    self.symb2int["fr"][symb] = cpt_symb_fr
                    cpt_symb_fr += 1
            data["fr"][data_cpt] = line

            # Anglais
            line = en.lower()
            line = re.split(r"(\W)", line)
            line = list(filter(lambda x: x not in symb_to_remove, line))
            for symb in line:
                if symb not in self.symb2int["en"]:
                    self.symb2int["en"][symb] = cpt_symb_en
                    cpt_symb_en += 1
            data["en"][data_cpt] = line
            data_cpt += 1
            if data_cpt >= n_samp:
                break

        # Dictionnaires d'entiers vers symboles
        self.int2symb = dict()
        self.int2symb["fr"] = {v: k for k, v in self.symb2int["fr"].items()}
        self.int2symb["en"] = {v: k for k, v in self.symb2int["en"].items()}

        # Ajout du padding pour les phrases francaises et anglaises
        self.max_len = dict()

        # ---------------------- Laboratoire 2 - Question 2 - Début de la section à compléter ------------------
        self.max_len["fr"] = max([len(item) for item in data["fr"].values()])
        self.max_len["en"] = max([len(item) for item in data["en"].values()])

        print(f"Longest word 'fr': {self.max_len['fr']}")
        print(f"Longest word 'en': {self.max_len['en']}")

        # Take <sos> into account
        self.max_len["fr"] += 1
        self.max_len["en"] += 1

        # Add padding
        for key in data["fr"].keys():
            data["fr"][key] = (
                data["fr"][key]
                + [self.stop_symbol]
                + [self.pad_symbol] * (self.max_len["fr"] - len(data["fr"][key]) - 1)
            )
        for key in data["en"].keys():
            data["en"][key] = (
                data["en"][key]
                + [self.stop_symbol]
                + [self.pad_symbol] * (self.max_len["en"] - len(data["en"][key]) - 1)
            )
        # ---------------------- Laboratoire 2 - Question 2 - Fin de la section à compléter ------------------

        # Assignation des données du dataset et de la taille du ditcionnaire
        self.data = data
        self.dict_size = {
            "fr": len(self.int2symb["fr"]),
            "en": len(self.int2symb["en"]),
        }

    def __len__(self):
        return len(self.data["fr"])

    def __getitem__(self, idx):
        fr_seq = self.data["fr"][idx]
        target_seq = self.data["en"][idx]
        fr_seq = [self.symb2int["fr"][i] for i in fr_seq]
        target_seq = [self.symb2int["en"][i] for i in target_seq]

        return torch.tensor(fr_seq), torch.tensor(target_seq)

    def visualize(self, idx):
        fr_seq, en_seq = [i.numpy() for i in self[idx]]
        fr_seq = [self.int2symb["fr"][i] for i in fr_seq]
        en_seq = [self.int2symb["en"][i] for i in en_seq]
        print("Français: ", " ".join(fr_seq))
        print("Anglais: ", " ".join(en_seq))


if __name__ == "__main__":
    print("\nExample de données de la base de données : \n")
    a = Fr_En("l2/fra.txt")
    a.visualize(np.random.randint(0, len(a)))
    print("\n")
