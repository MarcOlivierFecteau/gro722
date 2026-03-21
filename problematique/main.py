#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NOTES Consultation Problématique:
    - Distance de Levenshtein ciblée: 0.5, MAIS:
        - Pré-traitement à faire sur les données
            - Normalisation01: introduit autre problème;
            - Plutôt qu'utiliser les coordonnées absolues, on doit représenter les coordonnées comme une séquence de vecteurs entre deux points.
                - `np.diff(dim=-1)` => vecteur des coordonnées "relatives".
    - Padding des coordonnées avec vecteurs "relatifs": utiliser des zéros.
"""

import argparse
import os

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
        "-f",
        "--force_cpu",
        action=argparse.BooleanOptionalAction,
        help="Forcer l'utilisation du CPU",
    )
    parser.add_argument(
        "--training",
        action=argparse.BooleanOptionalAction,
        help="Entraîner le modèle",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        help="Tester le modèle",
    )
    parser.add_argument(
        "-c",
        "--learning_curves",
        action=argparse.BooleanOptionalAction,
        help="Afficher les courbes d'apprentissage",
    )
    parser.add_argument(
        "--gen_test_images",
        action=argparse.BooleanOptionalAction,
        help="Générer les images de test",
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
        default=1,
        help='Nombre de couches récurrentes "empilées" dans l\'architecture encodeur-décodeur',
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Charger le meilleur modèle"
    )
    parser.add_argument(
        "-a",
        "--attention",
        action=argparse.BooleanOptionalAction,
        help="Utiliser le module d'attention",
    )
    parser.add_argument(
        "--show_attention",
        action=argparse.BooleanOptionalAction,
        help="Affichage de l'attention",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        help="Utiliser une couche bidirectionnelle",
    )

    args = parser.parse_args()
    if args.seed == "None":
        seed = None
    else:
        seed = int(args.seed)
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
    if args.training:
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
            model = torch.load("problematique/best_model.pt", weights_only=False)
        else:
            model = Trajectory2Seq(
                args.num_hidden,
                args.num_layers,
                dataset.int2sym,
                dataset.sym2int,
                dataset.num_symbols,
                dataset.maxlength,
                device,
                attention_mod=args.attention,
                bidirectional=args.bidirectional,
            )
        print(
            f"Nombre de paramètres: {sum(param.numel() for param in model.parameters() if param.requires_grad)}"
        )
        print("-" * 30)

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
        criterion = nn.CrossEntropyLoss(
            ignore_index=dataset.sym2int[dataset.pad_symbol]
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        for epoch in tqdm(range(1, args.epochs + 1)):
            # @training
            running_train_distance = 0
            running_train_loss = 0.0

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

                outi = torch.argmax(out, dim=-1).detach().tolist()
                targeti = target.detach().tolist()
                for i in range(len(targeti)):
                    a = targeti[i]
                    b = outi[i]
                    stop_symbol = dataset.sym2int[dataset.stop_symbol]
                    targetlen = a.index(stop_symbol)
                    wordlen = b.index(stop_symbol) if stop_symbol in b else len(b)
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

            # @validation
            running_val_distance = 0
            running_val_loss = 0.0

            model.eval()
            for batch, data in enumerate(val_loader):
                coords, target = data
                coords = coords.float()
                target = target.long()
                out, hidden, attn = model(coords, target)
                logits = out.view(-1, model.num_symbols)
                target_ = target.view(-1)
                loss = criterion(logits, target_)
                running_val_loss += loss.item()

                outi = torch.argmax(out, dim=-1).detach().tolist()
                targeti = target.detach().tolist()
                for i in range(len(targeti)):
                    a = targeti[i]
                    b = outi[i]
                    stop_symbol = dataset.sym2int[dataset.stop_symbol]
                    targetlen = a.index(stop_symbol)
                    wordlen = b.index(stop_symbol) if stop_symbol in b else len(b)
                    running_val_distance += (
                        edit_distance(a[:targetlen], b[:wordlen]) / args.batch_size
                    )

            # Affichage
            coords, target = val_dataset[np.random.randint(0, len(val_dataset))]  # type: ignore
            target = target.unsqueeze(0).long()
            coords = coords.unsqueeze(0).float()
            out, hidden, attn = model(coords)
            out = torch.argmax(out, dim=-1).detach().squeeze(0).tolist()
            target = target.detach().squeeze(0).tolist()
            target_str = "".join([dataset.int2sym[i] for i in target if i != 2])
            out_str = "".join([dataset.int2sym[i] for i in out if i != 2])

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
                ax[0].set_title("Loss")  # type: ignore
                ax[1].set_title("Levenshtein Distance")  # type: ignore
                ax[0].set_xlabel("Epoch")  # type: ignore
                ax[0].set_ylabel("Loss")  # type: ignore
                ax[1].set_ylabel("Levenshtein Distance")  # type: ignore
                fig.tight_layout()  # type: ignore

                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            if epoch == 1 or (running_val_loss / len(val_loader)) <= min(val_loss):  # type: ignore
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
        plt.show(block=True)

    if args.test:
        # @test
        model = torch.load("problematique/best_model.pt", weights_only=False)
        model.to(device)
        model.eval()

        # Charger les données de tests
        test_dataset = HandwrittenWords("problematique/data_test.p")
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        pad_symbol = test_dataset.sym2int[test_dataset.pad_symbol]
        stop_symbol = test_dataset.sym2int[test_dataset.stop_symbol]
        start_symbol = test_dataset.sym2int[test_dataset.start_symbol]

        criterion = nn.CrossEntropyLoss(ignore_index=pad_symbol)

        print("-" * 30)
        print(f"Number of test samples: {len(test_dataset)}")
        print("-" * 30)

        def trim_at_eos(seq: list[int], eos_token: int, pad_token: int) -> list[int]:
            out = []
            for token in seq:
                if token == eos_token:
                    break
                out.append(token)
            return out

        def tokens_to_string(tokens: list[int], int2sym: dict[int, str]) -> str:
            return "".join(int2sym[t] for t in tokens)

        running_test_loss = 0.0
        running_distance = 0
        exact_matches = 0
        num_samples = 0

        all_truth_characters: list[str] = []
        all_prediction_characters: list[str] = []

        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                coords, target = data
                coords = coords.float().to(device)
                target = target.long().to(device)
                out, hidden, attn = model(coords, target)
                logits = out.view(-1, model.num_symbols)
                target_ = target.view(-1)
                loss = criterion(logits, target_)
                running_test_loss += loss.item()

                prediction = torch.argmax(out, dim=-1)

                prediction_list = prediction.cpu().tolist()
                target_list = target.cpu().tolist()

                for p_seq, t_seq in zip(prediction_list, target_list):
                    p_trimmed = trim_at_eos(p_seq, stop_symbol, pad_symbol)
                    t_trimmed = trim_at_eos(t_seq, stop_symbol, pad_symbol)
                    running_distance += edit_distance(p_trimmed, t_trimmed)
                    exact_matches += int(p_trimmed == t_trimmed)
                    num_samples += 1

                    # Align for character-level confusion matrix
                    common_length = min(len(p_trimmed), len(t_trimmed))
                    for i in range(common_length):
                        all_prediction_characters.append(
                            test_dataset.int2sym[p_trimmed[i]]
                        )
                        all_truth_characters.append(test_dataset.int2sym[t_trimmed[i]])

        avg_test_loss = running_test_loss / len(test_loader)
        avg_distance = running_distance / num_samples
        exact_match_accuracy = exact_matches / num_samples

        print("Test results:")
        print(f"Average loss: {avg_test_loss:.4f}")
        print(f"Average Levenshtein distance: {avg_distance:.4f}")
        print(f"Exact-match accuracy: {100 * exact_match_accuracy:.4f}%")

        # Display examples
        print("-" * 30)
        print("Examples")
        print("-" * 30)

        num_examples = min(5, len(test_dataset))
        with torch.no_grad():
            for _ in range(num_examples):
                i = np.random.randint(0, len(test_dataset))
                coords, target = test_dataset[i]
                abs_coords = test_dataset.original[i]

                coords = coords.unsqueeze(0).float().to(device)
                target = target.long().tolist()

                out, hidden, attn = model(coords)
                prediction = torch.argmax(out, dim=-1).squeeze(0).cpu().tolist()

                p_trimmed = trim_at_eos(prediction, stop_symbol, pad_symbol)
                t_trimmed = trim_at_eos(target, stop_symbol, pad_symbol)

                prediction_str = tokens_to_string(p_trimmed, test_dataset.int2sym)
                target_str = tokens_to_string(t_trimmed, test_dataset.int2sym)

                print(f"Target: {target_str:15s} | Prediction: {prediction_str}")

                if args.show_attention and attn is not None:
                    # TODO: display results using absolute coords, not relative vectors.
                    num_attn_steps = attn.size(
                        1
                    )  # Prevents indexing past available attention steps
                    wordlen = min(len(p_trimmed), num_attn_steps)

                    plt.figure(figsize=(4, max(2, wordlen)))
                    plt.suptitle(f"Attention Visualization\n(target: {target_str})")
                    for i in range(wordlen):
                        plt.subplot(wordlen, 1, i + 1)
                        attn_weights = attn[0, i, :].detach().cpu().numpy()
                        trajectory = abs_coords.detach().cpu().numpy()
                        plt.plot(trajectory[0], trajectory[1], "k", alpha=0.2)
                        plt.scatter(
                            trajectory[0],
                            trajectory[1],
                            c=1 - attn_weights,
                            cmap="grey",
                            s=10,
                        )
                        plt.title(test_dataset.int2sym[p_trimmed[i]])
                    plt.tight_layout()
                    plt.show(block=True)

        # Affichage de la matrice de confusion
        if len(all_truth_characters) > 0:
            confusion_mat = confusion_matrix(
                all_truth_characters,
                all_prediction_characters,
                ignore=[],
                show=True,
            )
