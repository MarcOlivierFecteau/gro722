# -*- coding: utf-8 -*-

import torch
from torch import Tensor, nn


class Trajectory2Seq(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        int2symb: dict[int, str],
        symb2int: dict[str, int],
        num_symbols: int,
        maxlen: dict[str, int],
        attention_mod: bool = False,
    ):
        super().__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.num_symbols = num_symbols
        self.maxlen = maxlen
        self.attention_mod = attention_mod

        # Definition des couches
        # Couches pour rnn
        # À compléter

        # Couches pour attention
        # À compléter

        # Couche dense pour la sortie
        # À compléter

    def forward(self, x):
        # À compléter
        return None
    

