#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import Tensor, nn


class Seq2seq(nn.Module):
    def __init__(
        self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len
    ):
        super(Seq2seq, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size["fr"], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size["en"], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size["en"])
        self.to(device)

    def encoder(self, x) -> tuple[Tensor, Tensor]:
        # Encodeur
        # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
        x = self.fr_embedding(x)
        out, hidden = self.encoder_layer(x, None)
        # ---------------------- Laboratoire 2 - Question 3 - Fin de la section à compléter -----------------

        return out, hidden

    def decoder(self, encoder_outs, hidden) -> tuple[Tensor, Tensor, Tensor | None]:
        # Initialisation des variables
        max_len = self.max_len[
            "en"
        ]  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = (
            torch.zeros((batch_size, 1)).to(self.device).long()
        )  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size["en"])).to(
            self.device
        )  # Vecteur de sortie du décodage

        # Boucle pour tous les symboles de sortie
        dec_hidden = hidden
        for i in range(max_len):
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------
            dec_out, dec_hidden = self.decoder_layer(
                self.en_embedding(vec_in), dec_hidden
            )
            dec_out = self.fc(dec_out)
            vec_out[:, i, :] = dec_out.squeeze(1)
            vec_in = dec_out.argmax(dim=-1)
            # ---------------------- Laboratoire 2 - Question 3 - Début de la section à compléter -----------------

        return vec_out, hidden, None

    def forward(self, x) -> tuple[Tensor, Tensor, Tensor | None]:
        # Passant avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoder(out, h)
        return out, hidden, attn


class Seq2seq_attn(nn.Module):
    def __init__(
        self, n_hidden, n_layers, int2symb, symb2int, dict_size, device, max_len
    ):
        super(Seq2seq_attn, self).__init__()

        # Definition des paramètres
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device
        self.symb2int = symb2int
        self.int2symb = int2symb
        self.dict_size = dict_size
        self.max_len = max_len

        # Définition des couches du rnn
        self.fr_embedding = nn.Embedding(self.dict_size["fr"], n_hidden)
        self.en_embedding = nn.Embedding(self.dict_size["en"], n_hidden)
        self.encoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)
        self.decoder_layer = nn.GRU(n_hidden, n_hidden, n_layers, batch_first=True)

        # Définition de la couche dense pour l'attention
        self.att_combine = nn.Linear(2 * n_hidden, n_hidden)
        self.hidden2query = nn.Linear(n_hidden, n_hidden)

        # Définition de la couche dense pour la sortie
        self.fc = nn.Linear(n_hidden, self.dict_size["en"])
        self.to(device)

    def encoder(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        x = self.fr_embedding(x)
        out, hidden = self.encoder_layer(x, None)
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return out, hidden

    def attentionModule(self, query: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        # Module d'attention

        # Couche dense à l'entrée du module d'attention
        query = self.hidden2query(query)

        # Attention

        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
        attention_weights = torch.zeros(
            (query.shape[0], query.shape[1], values.shape[1])
        )
        attention_output = torch.zeros(
            (query.shape[0], query.shape[1], values.shape[2])
        )

        for i in range(query.shape[0]):
            for j in range(query.shape[1]):
                for k in range(values.shape[1]):
                    attention_weights[i, j, k] = torch.dot(query[i, j], values[i, k])
                attention_weights[i, j] = nn.functional.softmax(
                    attention_weights[i, j], dim=-1
                )

        for i in range(query.shape[0]):
            for j in range(query.shape[1]):
                for k in range(values.shape[1]):
                    attention_output[i, j] += attention_weights[i, j, k] * values[i, k]
        # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return attention_output, attention_weights

    def decoderWithAttn(self, encoder_outs, hidden):
        # Décodeur avec attention

        # Initialisation des variables
        max_len = self.max_len[
            "en"
        ]  # Longueur max de la séquence anglaise (avec padding)
        batch_size = hidden.shape[1]  # Taille de la batch
        vec_in = (
            torch.zeros((batch_size, 1)).to(self.device).long()
        )  # Vecteur d'entrée pour décodage
        vec_out = torch.zeros((batch_size, max_len, self.dict_size["en"])).to(
            self.device
        )  # Vecteur de sortie du décodage
        attention_weights = torch.zeros(
            (batch_size, self.max_len["fr"], self.max_len["en"])
        ).to(self.device)  # Poids d'attention

        # Boucle pour tous les symboles de sortie
        for i in range(max_len):
            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------
            dec_out, hidden = self.decoder_layer(self.en_embedding(vec_in), hidden)
            attention_out, attention_weights = self.attentionModule(
                dec_out, encoder_outs
            )
            dec_out = self.fc(attention_out)
            vec_out[:, i, :] = dec_out.squeeze(1)
            vec_in = dec_out.argmax(dim=-1)
            # ---------------------- Laboratoire 2 - Question 4 - Début de la section à compléter -----------------

        return vec_out, hidden, attention_weights

    def forward(self, x):
        # Passe avant
        out, h = self.encoder(x)
        out, hidden, attn = self.decoderWithAttn(out, h)
        return out, hidden, attn
