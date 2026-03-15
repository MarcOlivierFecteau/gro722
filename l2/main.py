# GRO722 Laboratoire 2
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021

import matplotlib.pyplot as plt
import numpy as np
import torch

# Inclure la base de données, les modèles et les métriques
from dataset import Fr_En
from metrics import edit_distance
from models import Seq2seq, Seq2seq_attn
from torch import nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = 1  # Forcer l'utilisation du CPU (si un GPU est disponible)
    training = 1  # Faire l'entrainement sur l'ensemble de donnees
    display_attention = 0  # Affichage des poids d'attention
    learning_curves = 1  # Visualiser les courbes d'apprentissage pendant l'entrainement
    test = 1  # Visualiser la generation sur des echantillons de validation
    batch_size = 100  # Taille des lots
    n_epochs = 50  # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01  # Taux d'apprentissage pour l'optimizateur

    n_hidden = 20  # Nombre de neurones caches par couche
    n_layers = 2  # Nombre de de couches

    n_workers = 0  # Nombre de fils pour charger les donnees
    seed = None  # Pour repetabilite
    attention_mod: bool = False
    # ------------ Fin des paramètres et hyperparamètres ------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    )

    # Instanciation de l'ensemble de données
    dataset = Fr_En(n_samp=4000, samplelen=[6, 10])

    # Instanciation du dataloader
    dataload_train = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )

    print("Number of epochs : ", n_epochs)
    print("Training data : ", len(dataset))
    print("Taille dictionnaires: ", dataset.dict_size)
    print("\n")

    # Instanciation du model
    if attention_mod:
        model = Seq2seq_attn(
            n_hidden=n_hidden,
            n_layers=n_layers,
            device=device,
            symb2int=dataset.symb2int,
            int2symb=dataset.int2symb,
            dict_size=dataset.dict_size,
            max_len=dataset.max_len,
        )
    else:
        model = Seq2seq(
            n_hidden=n_hidden,
            n_layers=n_layers,
            device=device,
            symb2int=dataset.symb2int,
            int2symb=dataset.int2symb,
            dict_size=dataset.dict_size,
            max_len=dataset.max_len,
        )

    # Afficher le résumé du model
    print("Model : \n", model, "\n")
    print("Nombre de poids: ", sum([i.numel() for i in model.parameters()]))

    if training:
        # Initialisation affichage
        if learning_curves:
            train_dist = []  # Historique des distances
            train_loss = []  # Historique des coûts
            fig, ax = plt.subplots(1)  # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss(ignore_index=2)  # ignorer les symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            dist = 0
            for batch_idx, data in enumerate(dataload_train):
                # Formatage des données
                fr_seq, target_seq = data
                fr_seq = fr_seq.to(device).long()
                target_seq = target_seq.to(device).long()

                optimizer.zero_grad()  # Mise a zero du gradient
                output, hidden, attn = model(fr_seq)  # Passage avant
                loss = criterion(
                    output.view((-1, model.dict_size["en"])), target_seq.view(-1)
                )

                loss.backward()  # calcul du gradient
                optimizer.step()  # Mise a jour des poids
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(output, dim=-1).detach().cpu().tolist()
                target_seq_list = target_seq.cpu().tolist()
                M = len(output_list)
                for i in range(batch_size):
                    a = target_seq_list[i]
                    b = output_list[i]
                    Ma = a.index(1)  # longueur mot a
                    Mb = b.index(1) if 1 in b else len(b)  # longueur mot b
                    dist += edit_distance(a[:Ma], b[:Mb]) / batch_size

                # Affichage pendant l'entraînement
                print(
                    "Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}".format(
                        epoch,
                        n_epochs,
                        batch_idx * batch_size,
                        len(dataload_train.dataset),
                        100.0 * batch_idx * batch_size / len(dataload_train.dataset),
                        running_loss_train / (batch_idx + 1),
                        dist / len(dataload_train),
                    ),
                    end="\r",
                )

            print(
                "Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f} Average Edit Distance: {:.6f}".format(
                    epoch,
                    n_epochs,
                    (batch_idx + 1) * batch_size,
                    len(dataload_train.dataset),
                    100.0 * (batch_idx + 1) * batch_size / len(dataload_train.dataset),
                    running_loss_train / (batch_idx + 1),
                    dist / len(dataload_train),
                ),
                end="\r",
            )
            print("\n")
            # Affichage graphique
            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                train_dist.append(dist / len(dataload_train))
                ax.cla()
                ax.plot(train_loss, label="training loss")
                ax.plot(train_dist, label="training distance")
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            torch.save(model, "l2/model.pt")

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close("all")

    if test:
        # Évaluation

        # Chargement des poids
        model = torch.load("l2/model.pt")
        dataset.symb2int = model.symb2int
        dataset.int2symb = model.int2symb

        # Affichage des résultats
        for i in range(10):
            # Extraction d'une séquence du dataset de validation
            fr_seq, target_seq = dataset[np.random.randint(0, len(dataset))]

            # Évaluation de la séquence
            output, hidden, attn = model(torch.tensor(fr_seq)[None, :].to(device))
            out = torch.argmax(output, dim=2).detach().cpu()[0, :].tolist()

            # Affichage
            in_seq = [model.int2symb["fr"][i] for i in fr_seq.detach().cpu().tolist()]
            target = [
                model.int2symb["en"][i] for i in target_seq.detach().cpu().tolist()
            ]
            out_seq = [model.int2symb["en"][i] for i in out]

            out_seq = out_seq[: out_seq.index("<eos>") + 1]
            in_seq = in_seq[: in_seq.index("<eos>") + 1]
            target = target[: target.index("<eos>") + 1]

            print("Input:  ", " ".join(in_seq))
            print("Target: ", " ".join(target))
            print("Output: ", " ".join(out_seq))
            print("")
            if display_attention:
                attn = attn.detach().cpu()[0, :, :]
                plt.figure()
                plt.imshow(
                    attn[0 : len(in_seq), 0 : len(out_seq)],
                    origin="lower",
                    vmax=1,
                    vmin=0,
                    cmap="pink",
                )
                plt.xticks(np.arange(len(out_seq)), out_seq, rotation=45)
                plt.yticks(np.arange(len(in_seq)), in_seq)
                plt.show()
