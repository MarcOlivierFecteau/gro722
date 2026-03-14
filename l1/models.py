# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, n_hidden: int, n_layers: int = 1):
        super(Model, self).__init__()

        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        # RNN := (num_caract, h_size, n_layers)
        # self.rnn = nn.RNN(1, n_hidden, n_layers, batch_first=True)
        # self.rnn = nn.LSTM(1, n_hidden, n_layers, batch_first=True)
        self.rnn = nn.GRU(1, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------

    def forward(self, x, h=None, batch_first: bool = True):
        # ---------------------- Laboratoire 1 - Question 2, 6 - Début de la section à compléter ------------------
        if h is None:
            # x := (N, L, H_in)
            x, h = self.rnn(x)
        else:
            x, h = self.rnn(x, h)
        x = self.fc(x)
        x = torch.tanh(x)
        # ---------------------- Laboratoire 1 - Question 2, 6 - Fin de la section à compléter ------------------
        return x, h


if __name__ == "__main__":
    x = torch.zeros((100, 2, 1)).float()
    model = Model(25)
    print(model(x))
