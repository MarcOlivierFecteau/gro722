#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import HandwrittenWords
from metrics import confusion_matrix, edit_distance
from models import Trajectory2Seq
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

if __name__ == "__main__":
    # ---------------- Paramètres et hyperparamètres ----------------#
    parser = argparse.ArgumentParser(
        prog="GRO722",
        description="Réseaux de Neurones Récurrents - Problématique",
        exit_on_error=True,
    )
    parser.add_argument(
        "-f", "--force_cpu", action="store_true", help="Forcer l'utilisation du CPU"
    )
    parser.add_argument(
        "--training", type=bool, default=True, help="Entraîner le modèle", required=True
    )
    parser.add_argument("--test", type=bool, default=False, help="Tester le modèle")
    parser.add_argument(
        "-c",
        "--learning_curves",
        type=bool,
        default=True,
        help="Afficher les courbes d'apprentissage",
    )
    parser.add_argument(
        "--gen_test_images", action="store_true", help="Générer les images de test"
    )
    parser.add_argument("-s", "--seed", default="None", help="Pour répétabilité")
    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=0,
        help="Nombre de threads pour chargement des données",
    )
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=100, help="Taille des lots"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-2, help="Taux d'apprentissage"
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=20,
        help="Nombre de caches pour couches cachées",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Nombre de couches dans l'architecture encodeur-décodeur",
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Charger le meilleur modèle"
    )
    parser.add_argument(
        "--show_attention", action="store_true", help="Affichage de l'attention"
    )

    args = parser.parse_args()
    assert args.seed == "None" or isinstance(args.seed, int), (
        f"seed must be either 'None' or an integer (got: {args.seed})."
    )
    if args.seed == "None":
        seed = None
    else:
        seed = args.seed
    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix de l'appareil
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords("problematique/data_trainval.p")

    # Séparation de l'ensemble de données (entraînement et validation)
    num_samples = len(dataset)
    train_size = int(7 / 9 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Instanciation des dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print("-" * 30)
    print(f"Number of epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of samples in dataset: {len(dataset)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of symbols: {dataset.num_symbols}")
    print("-" * 30)

    # Instanciation du model
    if args.checkpoint and os.path.exists("problematique/best_model.pt"):
        model = torch.load("problematique/best_model.pt")
    else:
        model = Trajectory2Seq(
            args.num_hidden,
            args.num_layers,
            dataset.int2sym,
            dataset.sym2int,
            dataset.num_symbols,
            dataset.maxlength,
            attention_mod=False,
        )
    print(
        f"Nombre de paramètres: {sum(param.numel() for param in model.parameters() if param.requires_grad)}"
    )
    print("-" * 30)

    if args.training:
        # Initialisation affichage
        if args.learning_curves:
            train_distance: list[int] = []
            train_loss: list[float] = []
            val_distance: list[int] = []
            val_loss: list[float] = []
            fig, ax = plt.subplots(1, 2, sharex="row")
            ax[0].set_title("Loss")
            ax[1].set_title("Levenshtein Distance")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[1].set_ylabel("Levenshtein Distance")

        # Fonction de coût et optimizateur
        criterion = nn.CrossEntropyLoss()  # NOTE: ignorer symboles <pad>
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in tqdm(range(1, args.epochs + 1)):
            # Entraînement
            running_train_distance = 0
            running_train_loss = 0

            model.train()
            for batch, data in enumerate(tqdm(train_loader)):
                coords, target = data
                coords = coords.float()
                target = target.long()
                optimizer.zero_grad()
                out, hidden, attn = model(coords)
                out_ = out.view((-1, model.num_symbols))
                target_ = target.view(-1)
                loss = criterion(out_, target_)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

                outi = torch.argmax(out, dim=1).detach().tolist()
                targeti = target.detach().tolist()
                for i in range(len(targeti)):
                    a = targeti[i]
                    b = outi[i]
                    targetlen = a[1]
                    wordlen = b[1] if 1 in b else len(b)
                    running_train_distance += (
                        edit_distance(a[:targetlen], b[:wordlen]) / args.batch_size
                    )
                if batch % 100 == 0:
                    print(
                        f"Epoch {epoch},"
                        f"Batch {batch},"
                        f"Training loss: {running_train_loss / (batch + 1)},"
                        f"Levenshtein distance: {running_train_distance / (batch + 1)}"
                    )

            # Validation
            running_val_distance = 0
            running_val_loss = 0

            model.eval()
            breakpoint()
            for batch, data in enumerate(val_loader):
                coords, target = data
                coords = coords.float()
                target = target.long()
                out, hidden, attn = model(coords)
                target_1h = torch.zeros(
                    (args.batch_size, model.maxlen["target"], model.num_symbols)
                )
                target_1h = target_1h.scatter_(
                    2, target.view((args.batch_size, -1, 1)), 1
                )
                loss = criterion(out, target_1h)
                running_val_loss += loss.item()

                outi = torch.argmax(out, dim=-1).detach().tolist()
                targeti = target.detach().tolist()
                for i in range(len(targeti)):
                    a = targeti[i]
                    b = outi[i]
                    targetlen = a[1]
                    wordlen = b[1] if 1 in b else len(b)
                    running_val_distance += edit_distance(a[:targetlen], b[:wordlen])

            # Affichage
            target, coords = test_dataset[np.random.randint(0, len(test_dataset))]  # type: ignore
            target = target.unsqueeze(0).long()
            coords = coords.unsqueeze(0).float()
            out, hidden, attn = model(coords)
            out = torch.argmax(out, dim=-1).detach().squeeze(0).tolist()
            target = target.detach().squeeze(0).tolist()
            target_str = "".join([dataset.int2sym[i] for i in target if i != 2])
            out_str = "".join([dataset.int2sym[i] for i in target if i != 2])

            print(f"Target: {target_str}, Output: {out_str}")
            print(
                f"Epoch {epoch},"
                f"Validation loss: {running_val_loss / len(val_loader)},"
                f"Levenshtein distance: {running_val_distance / len(val_loader)}"
            )

            if args.learning_curves:
                train_distance.append(running_train_distance / len(train_loader))  # type: ignore
                train_loss.append(running_train_loss / len(train_loader))  # type: ignore
                val_distance.append(running_val_distance / len(val_loader))  # type: ignore
                val_loss.append(running_val_loss / len(val_loader))  # type: ignore
                ax[0].cla()  # type: ignore
                ax[0].plot(train_loss, label="training")  # type: ignore
                ax[0].plot(val_loss, label="validation")  # type: ignore
                ax[0].legend()  # type: ignore
                ax[1].cla()  # type: ignore
                ax[1].plot(train_distance, label="training")  # type: ignore
                ax[1].plot(val_distance, label="validation")  # type: ignore
                ax[1].legend()  # type: ignore
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            if epoch == 1 or (running_val_loss / len(val_loader)) < min(val_loss):  # type: ignore
                torch.save(model, "problematique/best_model.pt")
                with open("problematique/best_model.txt", "w") as f:
                    f.write(
                        f"Epoch: {epoch}, "
                        f"Training loss: {running_train_loss / len(train_loader)}, "
                        f"Levenshtein distance: {running_train_distance / len(train_loader)}, "
                        f"Validation loss: {running_val_loss / len(val_loader)}, "
                        f"Levenshtein distance: {running_val_distance / len(val_loader)}"
                    )
                    f.write(
                        "Config:\n\t"
                        f"num_hidden: {args.num_hidden}, "
                        f"num_layers: {args.num_layers}, "
                        f"learning_rate: {args.learning_rate}, "
                        f"epochs: {args.epochs}, "
                        f"batch_size: {args.batch_size}"
                    )
                    f.write(f"Criterion: {criterion}")
                    f.write(f"Optimizer: {optimizer}")
                    f.write(f"Model: {model}")

            torch.save(model, "problematique/last_model.pt")

    if args.test:
        # Évaluation
        model = torch.load("best_model.pt")
        model.eval()
        running_val_loss = 0
        distance = 0

        criterion = nn.CrossEntropyLoss(
            ignore_index=dataset.sym2int[dataset.pad_symbol]
        )  # Ignorer symboles <pad>

        # Charger les données de tests
        with open("problematique/data_testval.p", "rb") as fp:
            test_dataset = pickle.load(fp)
        test_loader = DataLoader(test_dataset)

        print("-" * 30)
        print(f"Number of test samples: {len(test_dataset)}")
        print("-" * 30)

        for batch, data in enumerate(test_loader):
            target, coords = data
            target = target.long()
            coords = coords.float()
            out, hidden, attn = model(coords)
            target_1h = torch.zeros(
                (args.batch_size, model.maxlen["target"], model.num_symbols)
            )
            target_1h = target_1h.scatter_(2, target.view((args.batch_size, -1, 1)), 1)
            loss = criterion(out.view((-1, model.num_symbols)), target.view(-1))
            running_val_loss += loss.item()
            outi = torch.argmax(out, dim=-1).detach().tolist()
            targeti = target.detach().tolist()
            for i in range(len(targeti)):
                a = targeti[i]
                b = outi[i]
                targetlen = a[1]
                wordlen = b[1] if 1 in b else len(b)
                distance += edit_distance(a[:targetlen], b[:wordlen])

        num_examples = 10
        for _ in range(num_examples):
            target, trajectory_seq = test_dataset[
                np.random.randint(0, len(test_dataset))
            ]

            out, hidden, attn = model(trajectory_seq.unsqueeze(0).float())
            out = torch.argmax(out, dim=-1).detach().squeeze(0).tolist()
            target = target.detach().tolist()

            target_str = "".join([model.int2sym[i] for i in target if i != 2])
            out_str = "".join([model.int2sym[i] for i in out if i != 2])
            print(f"Target: {target_str}, Output: {out_str}")

            # Affichage de l'attention
            if args.show_attention:
                wordlen = len(out)
                plt.figure(figsize=(1, 1 * wordlen))
                for i in range(wordlen):
                    plt.subplot(wordlen, 1, i + 1)
                    attn_weights = attn[0, i, :].detach().numpy()
                    trajectory = trajectory_seq.detach().numpy()
                    plt.scatter(
                        trajectory[0], trajectory[1], c=attn_weights, cmap="grey", s=10
                    )
                    plt.title(dataset.int2sym[out[i]])
                plt.show(block=True)

        # Affichage des résultats de test
        # À compléter

        # Affichage de la matrice de confusion
        # À compléter

        pass
