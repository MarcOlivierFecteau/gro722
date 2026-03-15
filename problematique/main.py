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
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True           # Entrainement?
    test = True                # Test?
    learning_curves = True     # Affichage des courbes d'entrainement?
    gen_test_images = True     # Génération images test?
    seed = 1                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    # À compléter
    n_epochs = 0

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    # À compléter

    
    # Séparation de l'ensemble de données (entraînement et validation)
    # À compléter
   

    # Instanciation des dataloaders
    # À compléter


    # Instanciation du model
    # À compléter


    # Initialisation des variables
    # À compléter

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