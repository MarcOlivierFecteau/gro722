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

        # Définition des couches
        # Couches pour RNN
        self.enc = nn.GRU(2, self.hidden_dim, self.n_layers, batch_first=True)
        self.dec = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.embed = nn.Embedding(self.num_symbols, self.hidden_dim)

        # Couches pour attention
        self.attn_ff = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.num_symbols)

        self._decoder = self.decoder_with_attention if attention_mod else self.decoder

