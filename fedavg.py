"""
Exemple pédagogique : Federated Learning avec Flower et FedAvg sur MNIST

Objectif :
- Créer plusieurs clients simulés
- Chaque client entraîne localement un modèle sur ses propres données
- Le serveur agrège les modèles avec FedAvg
- Le modèle global est évalué après chaque round
"""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import tensorflow as tf

from flwr.common import Metrics
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth


# ============================================================
# 1. Paramètres globaux
# ============================================================

VERBOSE = 0

NUM_CLIENTS = 5
NUM_ROUNDS = 10

FRACTION_FIT = 0.1
FRACTION_EVALUATE = 0.1

BATCH_SIZE = 32
LOCAL_EPOCHS = 5

TOTAL_DATASET_SIZE = 60000


# ============================================================
# 2. Définition du modèle local
# ============================================================

def get_model() -> tf.keras.Model:
    """
    Crée un modèle simple pour MNIST.

    MNIST contient des images 28x28 représentant des chiffres de 0 à 9.
    Le modèle prend une image en entrée et prédit la classe correspondante.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 3. Définition d’un client Flower
# ============================================================

class FlowerClient(fl.client.NumPyClient):
    """
    Représente un client dans le système fédéré.

    Chaque client possède :
    - son propre modèle local ;
    - son propre ensemble d'entraînement ;
    - son propre ensemble de validation.

    Le client reçoit le modèle global du serveur,
    l'entraîne localement, puis renvoie ses poids mis à jour.
    """

    def __init__(self, trainset, valset, cid: str) -> None:
        self.cid = cid
        self.model = get_model()

        self.trainset = trainset.to_tf_dataset(
            columns="image",
            label_cols="label",
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        self.valset = valset.to_tf_dataset(
            columns="image",
            label_cols="label",
            batch_size=BATCH_SIZE,
        )

    def get_parameters(self, config):
        """
        Retourne les paramètres actuels du modèle local.

        Ces paramètres correspondent aux poids du réseau de neurones.
        """
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Entraînement local du client.

        Étapes :
        1. Le client reçoit les paramètres du modèle global.
        2. Il met à jour son modèle local avec ces paramètres.
        3. Il entraîne le modèle sur ses propres données locales.
        4. Il renvoie les nouveaux paramètres au serveur.
        """

        self.model.set_weights(parameters)

        self.model.fit(
            self.trainset,
            epochs=LOCAL_EPOCHS,
            verbose=VERBOSE,
        )

        return self.model.get_weights(), len(self.trainset), {}

    def evaluate(self, parameters, config):
        """
        Évaluation locale du modèle.

        Le client reçoit le modèle global, l'évalue sur ses données
        de validation locales, puis renvoie la loss et l'accuracy.
        """

        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(
            self.valset,
            verbose=VERBOSE,
        )

        return loss, len(self.valset), {"accuracy": accuracy}


# ============================================================
# 4. Création des clients
# ============================================================

def get_client_fn(dataset: FederatedDataset):
    """
    Retourne une fonction permettant à Flower de créer un client.

    Flower appelle cette fonction automatiquement pour créer les clients
    nécessaires pendant la simulation.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """
        Crée un client à partir de son identifiant cid.

        Chaque client reçoit une partition différente du dataset.
        """

        client_dataset = dataset.load_partition(int(cid), "train")

        client_dataset_splits = client_dataset.train_test_split(
            test_size=0.1
        )

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        return FlowerClient(trainset, valset, cid).to_client()

    return client_fn


# ============================================================
# 5. Agrégation des métriques d’évaluation
# ============================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Agrège les accuracy retournées par les clients.

    Chaque accuracy est pondérée par le nombre d'exemples du client.
    Cela évite de donner le même poids à un petit client et à un grand client.
    """

    accuracies = [
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ]

    examples = [
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


# ============================================================
# 6. Évaluation centralisée côté serveur
# ============================================================

def get_evaluate_fn(testset):
    """
    Crée une fonction d'évaluation côté serveur.

    Après chaque round, le serveur peut évaluer le modèle global
    sur un jeu de test centralisé.
    """

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()
        model.set_weights(parameters)

        loss, accuracy = model.evaluate(
            testset,
            verbose=VERBOSE,
        )

        print(
            f"Round {server_round} - "
            f"Server-side loss: {loss:.4f}, "
            f"accuracy: {accuracy:.4f}"
        )

        return loss, {"accuracy": accuracy}

    return evaluate


# ============================================================
# 7. Préparation du dataset fédéré
# ============================================================

# Désactiver l'utilisation directe du GPU dans le processus principal
# pour éviter certains conflits avec Ray/Flower en simulation.
tf.config.experimental.set_visible_devices([], "GPU")

min_partition_size = TOTAL_DATASET_SIZE // NUM_CLIENTS // 10

partitioner = DirichletPartitioner(
    num_partitions=NUM_CLIENTS,
    partition_by="label",
    alpha=0.5,
    min_partition_size=min_partition_size,
    self_balancing=True,
)

mnist_fds = FederatedDataset(
    dataset="mnist",
    partitioners={"train": partitioner},
)

centralized_testset = mnist_fds.load_split("test").to_tf_dataset(
    columns="image",
    label_cols="label",
    batch_size=BATCH_SIZE,
)


# ============================================================
# 8. Définition de la stratégie FedAvg
# ============================================================

strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,

    min_fit_clients=max(1, int(NUM_CLIENTS * FRACTION_FIT)),
    min_evaluate_clients=max(1, int(NUM_CLIENTS * FRACTION_EVALUATE)),
    min_available_clients=NUM_CLIENTS,

    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(centralized_testset),
)


# ============================================================
# 9. Ressources utilisées par chaque client simulé
# ============================================================

client_resources = {
    "num_cpus": 1,
    "num_gpus": 1,
}


# ============================================================
# 10. Lancement de la simulation fédérée
# ============================================================

history = fl.simulation.start_simulation(
    client_fn=get_client_fn(mnist_fds),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,
    actor_kwargs={
        "on_actor_init_fn": enable_tf_gpu_growth,
    },
)