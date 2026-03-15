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

    if trainning:

        # Fonction de coût et optimizateur
        # À compléter

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            # À compléter
            
            # Validation
            # À compléter

            # Ajouter les loss aux listes
            # À compléter

            # Enregistrer les poids
            # À compléter


            # Affichage
            if learning_curves:
                # visualization
                # À compléter
                pass

    if test:
        # Évaluation
        # À compléter

        # Charger les données de tests
        # À compléter

        # Affichage de l'attention
        # À compléter (si nécessaire)

        # Affichage des résultats de test
        # À compléter

        # Affichage de la matrice de confusion
        # À compléter

        pass
