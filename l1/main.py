# GRO722 Laboratoire 1
# Auteurs: Jean-Samuel Lauzon et Jonathan Vincent
# Hiver 2021
import matplotlib.pyplot as plt
import numpy as np
import torch

# Inclure la base de donnéess
from dataset import SignauxDataset

# Inclure le modèle
from models import Model
from torch import nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = True  # Forcer l'utilisation du CPU (si un GPU est disponible)
    training = True  # Faire l'entrainement sur l'ensemble de donnees
    learning_curves = (
        True  # Visualiser les courbes d'apprentissage pendant l'entrainement
    )
    test_tagging = True  # Visualiser l'annotation sur des echantillons de validation
    test_generation = (
        True  # Visualiser la generation sur des echantillons de validation
    )

    batch_size = 10  # Taille des lots
    n_epochs = 50  # Nombre d'iteration sur l'ensemble de donnees
    lr = 0.01  # Taux d'apprentissage pour l'optimizateur

    n_hidden = 25  # Nombre de neurones caches par couche
    n_layers = 1  # Nombre de de couches

    n_workers = 0  # Nombre de fils pour charger les donnees
    train_val_split = 0.7  # Ratio des echantillions pour l'entrainement
    seed = None  # Pour repetabilite
    # ------------ Fin des paramètres et hyperparamètres -----------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Choix du device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    )

    # Instanciation de l'ensemble de données
    dataset = SignauxDataset()

    # Séparation du dataset (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset, [n_train_samp, n_val_samp]
    )

    # Instanciation des dataloaders
    dataload_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers
    )
    dataload_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers
    )

    print("Number of epochs : ", n_epochs)
    print("Training data : ", len(dataset_train))
    print("Validation data : ", len(dataset_val))
    print("\n")

    # Instanciation du model
    model = Model(n_hidden, n_layers=n_layers)
    model = model.to(device)

    # Afficher le résumé du model
    print("Model : \n", model, "\n")

    # Initialisation des variables
    best_val_loss = np.inf  # pour sauvegarder le meilleur model

    if training:
        # Initialisation affichage
        if learning_curves:
            val_loss = []  # Historique des coûts
            train_loss = []  # Historique des coûts
            fig, ax = plt.subplots(1)  # Initialisation figure

        # Fonction de coût et optimizateur
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            model.train()
            for batch_idx, data in enumerate(dataload_train):
                in_seq, target_seq = [obj.to(device).float() for obj in data]

                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
                in_seq = torch.unsqueeze(in_seq, 2)
                target_seq = torch.unsqueeze(target_seq, 2)
                optimizer.zero_grad()
                out_seq, h = model.forward(in_seq)
                loss = criterion(out_seq, target_seq)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()
                # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter ------------------

                # Affichage pendant l'entraînement
                if batch_idx % 10 == 0:
                    print(
                        "Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}".format(
                            epoch,
                            n_epochs,
                            batch_idx * len(data),
                            len(dataload_train.dataset),
                            100.0 * batch_idx / len(dataload_train),
                            running_loss_train / (batch_idx + 1),
                        ),
                        end="\r",
                    )

            # Validation
            running_loss_val = 0
            model.eval()
            for data in dataload_val:
                in_seq, target_seq = [obj.to(device).float() for obj in data]

                # ---------------------- Laboratoire 1 - Question 3 - Début de la section à compléter ------------------
                in_seq = torch.unsqueeze(in_seq, 2)
                target_seq = torch.unsqueeze(target_seq, 2)
                out_seq, _ = model.forward(in_seq)
                loss = criterion(out_seq, target_seq)
                running_loss_val = loss.item() * in_seq.size(0)
                # ---------------------- Laboratoire 1 - Question 3 - Fin de la section à compléter ------------------

            print(
                "\nValidation - Average loss: {:.4f}".format(
                    running_loss_val / len(dataload_val)
                )
            )
            print("")

            # Affichage
            if learning_curves:
                train_loss.append(running_loss_train / len(dataload_train))
                val_loss.append(running_loss_val / len(dataload_val))
                ax.cla()
                ax.plot(train_loss, label="training loss")
                ax.plot(val_loss, label="validation loss")
                ax.legend()
                plt.draw()
                plt.pause(0.01)

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, "model.pt")

        # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close("all")

    if test_tagging:
        # Évaluation étiquettage
        model = torch.load(
            "model.pt", map_location=lambda storage, loc: storage, weights_only=False
        )
        model = model.to(device)
        model.eval()
        for num in range(10):
            # Extraction d'une séquence du dataset de validation
            input_sequence, target_sequence = dataset_val[
                np.random.randint(0, len(dataset_val))
            ]

            # Initialisation de la prédiction de sortie
            prediction_sequence = np.zeros(len(input_sequence))

            # ---------------------- Laboratoire 1 - Question 4 - Début de la section à compléter ------------------
            input_sequence = torch.unsqueeze(input_sequence, 1).to(torch.float32)
            target_sequence = torch.unsqueeze(target_sequence, 1).to(torch.float32)
            prediction_sequence, _ = model.forward(input_sequence)
            # ---------------------- Laboratoire 1 - Question 4 - Fin de la section à compléter ------------------

            plt.title("Tagged data")
            plt.plot(target_sequence, label="target")
            plt.plot(prediction_sequence.detach().numpy(), label="prediction")
            plt.legend()
            plt.show()

    if test_generation:
        # Évaluation génération
        model = torch.load(
            "model.pt", map_location=lambda storage, loc: storage, weights_only=False
        )
        model = model.to(device)
        model.eval()
        for num in range(10):
            # Extraction d'une séquence du dataset de validation
            input_sequence, target_sequence = dataset_val[
                np.random.randint(0, len(dataset_val))
            ]

            # Calcul du nombre de prédictions à générer
            usable_input_sequence_len = len(input_sequence) >> 1
            nb_predictions_to_generate = len(input_sequence) - usable_input_sequence_len

            # Initialisation de la prédiction de sortie
            prediction_sequence = np.zeros(nb_predictions_to_generate)

            # ---------------------- Laboratoire 1 - Question 5 - Début de la section à compléter ------------------
            with torch.no_grad():
                # Annotation sur la première moitié de la séquence.
                annotated_input = (
                    input_sequence[:usable_input_sequence_len]
                    .unsqueeze(1)
                    .to(torch.float32)
                )
                annotated_output, h = model.forward(annotated_input)

                # Conserver la dernière sortie et le dernier contexte pour initier la génération.
                current_input = annotated_output[-1].view(1, 1)

                # Génération auto-régressive: une sortie à la fois.
                for pred_idx in range(nb_predictions_to_generate):
                    generated_output, h = model.forward(current_input, h)
                    prediction_sequence[pred_idx] = generated_output[-1, 0].item()
                    current_input = generated_output[-1].view(1, 1)
            # ---------------------- Laboratoire 1 - Question 5 - Fin de la section à compléter ------------------

            prediction_t = [
                i + usable_input_sequence_len for i in range(nb_predictions_to_generate)
            ]
            plt.plot(target_sequence)
            plt.plot(prediction_t, prediction_sequence)
            plt.title("Generated data")
            plt.show()
