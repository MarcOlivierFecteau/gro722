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

    def encoder(self, x: Tensor):
        """
        Args:
            x (Tensor): Input sequence (array[(x, y)]).

        Returns:
            out (Tensor): Output sequence.
            hidden (Tensor): Hidden layer.
        """
        x = x.permute(
            0, 2, 1
        )  # Equivalent of transpose applied on all samples of batch
        out, hidden = self.enc(x, None)
        return out, hidden

    def decoder(self, enc_out: Tensor, hidden: Tensor, target: Tensor | None = None):
        """
        Args:
            enc_out (Tensor): Encoder output (h[N-1]).
            hidden (Tensor): Decoder hidden layer (from previous layer).
            target (Tensor): Tokenized target (y).

        Returns:
            v_out (Tensor): Decoder output.
            hidden (Tensor): Decoder hidden layer (updated).
        """
        maxlen = self.maxlen["target"]
        batch_size = hidden.shape[1]
        v_in = torch.zeros((batch_size, 1)).long()
        v_out = torch.zeros((batch_size, maxlen, self.num_symbols))

        dec_hidden = hidden
        for i in range(maxlen):
            dec_out, dec_hidden = self.dec(self.embed(v_in), dec_hidden)
            dec_out = self.fc(dec_out)
            v_out[:, i, :] = dec_out.squeeze(1)
            v_in = torch.argmax(dec_out, dim=2)

        return v_out, hidden, None

    def attention(self, values: Tensor, query: Tensor):
        """
        Compute similiarity, softmax and weighting functions in attention module.

        Args:
            values (Tensor): Encoder output (v[0..N-1]).
            query (Tensor): Decoder output (q[0..M-1]).

        Returns:
            attn_out (Tensor): Attention output (a[0..M-1]).
            attn_weights (Tensor): Internal weights (w).
        """
        query = self.query(query)
        similarity = torch.bmm(values, query.permute(0, 2, 1)).squeeze(-1)
        attn_weights = nn.functional.softmax(similarity, dim=-1).unsqueeze(1)
        attn_out = torch.bmm(attn_weights, values)
        return attn_out, attn_weights

    def decoder_with_attention(
        self, enc_out: Tensor, hidden: Tensor, target: Tensor | None = None
    ):
        """
        Args:
            enc_out (Tensor): Encoder output (h[N-1]).
            hidden (Tensor): Decoder hidden layer (from previous layer).
            target (Tensor): Tokenized target (y).

        Returns:
            v_out (Tensor): Decoder output.
            hidden (Tensor): Decoder hidden layer (updated).
            attn_weights (Tensor): Attention module internal weights (w).
        """
        maxlen = self.maxlen["target"]
        batch_size = hidden.shape[1]
        v_in = torch.zeros((batch_size, 1)).long()
        v_out = torch.zeros((batch_size, maxlen, self.num_symbols))
        attn_weights = torch.zeros((batch_size, self.maxlen["coords"], maxlen))

        for i in range(maxlen):
            query, hidden, _ = self.dec(self.embed(v_in), hidden)
            attn_out, attn_weights = self.attention(query, enc_out)
            attn_weights[:, :, i] = attn_weights.squeeze(1)
            ff_out = self.attn_ff(torch.cat((query, attn_out), dim=-1))
            dec_out = self.fc(ff_out)

            v_out[:, i, :] = dec_out.squeeze(1)
            v_in = dec_out.argmax(dim=2)

        return v_out, hidden, attn_weights

    def forward(self, x: Tensor, target: Tensor | None = None):
        out, hidden = self.encoder(x)
        out, hidden, attn = self._decoder(out, hidden, target)
        return out, hidden, attn
