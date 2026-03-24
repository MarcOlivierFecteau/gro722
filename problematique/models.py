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
        device: torch.types.Device,
        attention_mod: bool = False,
        bidirectional: bool = False,
        teach_threshold: float = 0.5,
    ):
        super().__init__()
        # Definition des parametres
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sym2int = symb2int
        self.int2sym = int2symb
        self.num_symbols = num_symbols
        self.maxlen = maxlen
        self.attention_mod = attention_mod
        self.bidirectional = bidirectional
        self.teach_threshold = teach_threshold

        # Définition des couches
        # Couches pour RNN
        self.encoder_layer = nn.GRU(
            2, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.decoder_layer = nn.GRU(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.dec_embedding_layer = nn.Embedding(num_symbols, hidden_dim)

        # Couches pour attention
        self.attn_ff = nn.Linear(2 * hidden_dim, hidden_dim)
        # self.hidden_to_query = nn.Linear(hidden_dim, hidden_dim)  # Futile

        if bidirectional:
            self.hidden_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
            self.enc_out_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.hidden_bridge = nn.Identity()
            self.enc_out_bridge = nn.Identity()

        # Couche dense pour la sortie
        self.fc = nn.Linear(self.hidden_dim, self.num_symbols)

        self._decoder = self.decoder_with_attention if attention_mod else self.decoder

    def _bridge_hidden(self, hidden: Tensor) -> Tensor:
        """(2L, N, H) -> (L, N, H)"""
        if not self.bidirectional:
            return hidden

        # hidden := (2L, N, H) -> (L, 2, N, H)
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), self.hidden_dim)

        # concat forward/backward: (L, N, 2H)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)

        # Project back to decoder size: (L, N, H)
        hidden = self.hidden_bridge(hidden)
        return hidden

    def _bridge_enc_out(self, enc_out: Tensor) -> Tensor:
        """(N, L, 2H) -> (N, L, H)"""
        if not self.bidirectional:
            return enc_out
        return self.enc_out_bridge(enc_out)

    def encoder(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): Input sequence (array[(x, y)]).

        Returns:
            out (Tensor): Output sequence (N, L, H).
            hidden (Tensor): Hidden layer (N, L, H).
        """
        x = x.permute(0, 2, 1)  # Match dimensions
        out, hidden = self.encoder_layer(x, None)
        out = self._bridge_enc_out(out)
        hidden = self._bridge_hidden(hidden)
        return out, hidden

    def decoder(
        self, enc_out: Tensor, hidden: Tensor, target: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Args:
            enc_out (Tensor): Encoder output (v). Unused: exists to keep consistent signature.
            hidden (Tensor): Decoder hidden layer (h).
            target (Tensor): Tokenized target (y).

        Returns:
            v_out (Tensor): Decoder output (yhat).
            hidden (Tensor): Decoder hidden layer (updated).
        """
        device = hidden.device
        maxlen = self.maxlen["target"]
        batch_size = hidden.shape[1]
        v_in = torch.full((batch_size, 1), self.sym2int["<sos>"], device=device).long()
        v_out = torch.zeros((batch_size, maxlen, self.num_symbols), device=device)

        dec_hidden = hidden
        teach_next = False
        for i in range(maxlen):
            dec_out, dec_hidden = self.decoder_layer(
                self.dec_embedding_layer(v_in), dec_hidden
            )
            dec_out = self.fc(dec_out)
            if teach_next:
                v_out[:, i, :] = target[i]  # type: ignore
            else:
                v_out[:, i, :] = dec_out.squeeze(1)
            v_in = torch.argmax(dec_out, dim=2)
            teach_next = torch.rand((1,)).tolist()[0] > self.teach_threshold

        return v_out, dec_hidden, None

    def attention(self, values: Tensor, query: Tensor):
        """
        Compute similiarity, softmax and weighting functions in attention module.

        Args:
            values (Tensor): Encoder output (N, L_src, H).
            query (Tensor): Decoder output (N, 1, H).

        Returns:
            attn_out (Tensor): Attention output (N, 1, H).
            attn_weights (Tensor): Internal weights (N, 1, L_src).
        """
        # query = self.hidden_to_query(query)  # Optional
        similarity = torch.bmm(query, values.permute(0, 2, 1))
        attn_weights = nn.functional.softmax(similarity, dim=-1)
        attn_out = torch.bmm(attn_weights, values)
        return attn_out, attn_weights

    def decoder_with_attention(
        self, enc_out: Tensor, hidden: Tensor, target: Tensor | None = None
    ):
        """
        Args:
            enc_out (Tensor): Encoder output (N, L_src, H).
            hidden (Tensor): Decoder hidden layer (1, N, H).
            target (Tensor): Tokenized target (N, L_target).

        Returns:
            v_out (Tensor): Decoder output (N, L_target, num_symbols).
            hidden (Tensor): Decoder hidden layer (1, N, H).
            all_attn_weights (Tensor): Attention module internal weights (N, L_target, L_src).
        """
        device = hidden.device
        maxlen = self.maxlen["target"]
        batch_size = hidden.shape[1]
        src_length = enc_out.size(1)
        start_symbol = self.sym2int["<sos>"]

        v_in = torch.full(
            (batch_size, 1), start_symbol, device=device, dtype=torch.long
        )
        v_out = torch.zeros((batch_size, maxlen, self.num_symbols), device=device)
        all_attn_weights = torch.zeros((batch_size, maxlen, src_length), device=device)

        dec_hidden = hidden
        for i in range(maxlen):
            query, dec_hidden = self.decoder_layer(
                self.dec_embedding_layer(v_in), dec_hidden
            )  # query := (N, 1, H), dec_hidden := (1, N, H)
            attn_out, attn_weights = self.attention(
                enc_out, query
            )  # attn_out := (N, 1, H), attn_weights := (N, 1, L_src)
            ff_out = self.attn_ff(
                torch.cat((query.squeeze(1), attn_out.squeeze(1)), dim=1)
            )
            dec_out = self.fc(ff_out)  # (N, num_symbols)
            v_out[:, i, :] = dec_out
            all_attn_weights[:, i, :] = attn_weights.squeeze(1)

            if target is not None and i < target.size(1):
                v_in = target[:, i].unsqueeze(1)
            else:
                v_in = torch.argmax(dec_out, dim=-1, keepdim=True)

        return v_out, dec_hidden, all_attn_weights

    def forward(self, x: Tensor, target: Tensor | None = None):
        out, hidden = self.encoder(x)  # out := (N, L_src, H), hidden := (1, N, H)
        out, hidden, attn = self._decoder(out, hidden, target)
        return out, hidden, attn
