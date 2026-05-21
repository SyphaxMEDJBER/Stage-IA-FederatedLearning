"""
Exemple pédagogique : Federated Learning avec Flower et FedAvg sur MNIST

Objectif :
- Créer plusieurs clients simulés
- Chaque client entraîne localement un modèle sur ses propres données
- Le serveur agrège les modèles avec FedAvg
- Le modèle global est évalué après chaque round
"""

from typing import Dict, List, Tuple

import flwr as fl #gère le federated learning
import numpy as np # sert à manipuler des tableaux numériques et les poids du modèle
import tensorflow as tf # sert à créer et entrainer le modèle ia 

from flwr.common import Metrics # le type Metrics utilisé pour stocker les métriques comme accuracy ou loss 
from flwr_datasets import FederatedDataset #une classe permettant de changer un dataset déjà existant
from flwr_datasets.partitioner import DirichletPartitioner #pour repartir les données entre plusieurs clients de maniere aléatoire et unifi
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growthévite    #les problèmes liés à l’utilisation du GPU avec TensorFlow pendant les simulations Flower.


# ============================================================
# 1. Paramètres globaux
# ============================================================

VERBOSE = 0 #desactive l'affichage détaillé de tenserflow pendant l'entrainement

NUM_CLIENTS = 5  # nombre de client dans la simulation
NUM_ROUNDS = 10 #noombre de cycle

FRACTION_FIT = 0.1 # pourcentage des clients utilisé pour le l'entrainement à chaque round (ex: 10%  des rounds)
FRACTION_EVALUATE = 0.1 #pourcentage de client utilisé pour l'évaluation de modele 

BATCH_SIZE = 32 # nombre d'images envoyées au modele a chaque etape d'entrainement (ici le modele traite 32 images a la fois )
LOCAL_EPOCHS = 5 #nombre de fois ou chaque client entraine localement ses données

TOTAL_DATASET_SIZE = 60000 # taille de dataset utilisé pour la simulation


# ============================================================
# 2. Définition du modèle local
# ============================================================

def get_model() -> tf.keras.Model: # fonction pour créer le modele
    """
    Crée un modèle simple pour MNIST.

    MNIST contient des images 28x28 représentant des chiffres de 0 à 9.
    Le modèle prend une image en entrée et prédit la classe correspondante.
    """

    model = tf.keras.Sequential([  #création du modèle composé de plusieurs couches
        tf.keras.layers.Input(shape=(28, 28)),# le model recoit une image MNIST de taile 28*28 pixels
        tf.keras.layers.Flatten(),#transforme l'image en un seul ligne de nombre 28*28=784 valuurs
        tf.keras.layers.Dense(128, activation="relu"),#couche de 128 neurones qui apprend les caractéristiques 
        #importantes des images,fonction dactivation 
        tf.keras.layers.Dropout(0.2),#désactive 20% des neurones pendant l'entrainement pour eviter d'apprendre par coeur
        tf.keras.layers.Dense(10, activation="softmax"),#couche finale avec 10 sorties car il y a 10 chiffres possibles
    ])

    model.compile(   #prepare le modele pour lentrainement 
        optimizer="adam",  #algo qui ameliore le modele pendant lapprentissage
        loss="sparse_categorical_crossentropy", # la fonction qui mesure les erreurs du modèle
        metrics=["accuracy"],# demande d'afficher la precision de modèle
    )

    return model # retourne le modèle pret a etre utilisé par les clients  


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

    def __init__(self, trainset, valset, cid: str) -> None:# initialisation des champs  du client flower
        self.cid = cid #id du client 
        self.model = get_model()# chaque client crée sa propre copie locale du modèle IA

        self.trainset = trainset.to_tf_dataset( #ransforme les données d’entraînement en dataset TensorFlow utilisable par le modèle
            columns="image", #indique que les images sont les données d'entrée
            label_cols="label", #indique les bonnes réponses attendus
            batch_size=BATCH_SIZE, # envoie les images par groupes de 32 pendant l'entrainement
            shuffle=True, # mélange les données aleatoirement avant l'netrainement, ca ameliore souvent l'apprentissage
        )

        self.valset = valset.to_tf_dataset(#prépare aussi le dataset de validation TensorFlow
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

        self.model.set_weights(parameters)#le client recoit les param du modèle global et met à jour les siens

        self.model.fit( #lance l'entrainement du modèle ia local du client
            self.trainset, #utilise la dataset d'entrainement du client
            epochs=LOCAL_EPOCHS,#nombre de repition de client sur les donnés 
            verbose=VERBOSE, #controle l'affichage des details tenserFlow , ici = 0 donc détails désactivés
        )

        return self.model.get_weights(), len(self.trainset), {}#le client renvoie ls poids et le nombre de données utilisées par le client pour lentrainement 

    def evaluate(self, parameters, config):
        """
        Évaluation locale du modèle.

        Le client reçoit le modèle global, l'évalue sur ses données
        de validation locales, puis renvoie la loss et l'accuracy.
        """

        self.model.set_weights(parameters) # mettre a jour les parametres de son model local avec celles reçues de serveur  

        loss, accuracy = self.model.evaluate(# lancer l'evaluation du modèle 
            self.valset,
            verbose=VERBOSE,
        )

        return loss, len(self.valset), {"accuracy": accuracy} #envoie au serveur la loss et le nombre de données utilisées 


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

        client_dataset = dataset.load_partition(int(cid), "train")#charge la partie du dataset apparetenant à ce client 

        client_dataset_splits = client_dataset.train_test_split( #sépare les données du client ,entrainement/validation  
            test_size=0.1 # 10% pour la validation et 90% pour lentrainement 
        )

        trainset = client_dataset_splits["train"]#données utilisées pour entrainer le modèle
        valset = client_dataset_splits["test"]#données utilisées pour validation

        return FlowerClient(trainset, valset, cid).to_client() #creation du client (appel au constructeur)
        #to_client pour transformer  l'objet client a un format special compatible avec le systeme de 
        # fLOWER 

    return client_fn  # retourne la fonction de creation des clients 


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
        num_examples * metric["accuracy"] # on multiplie chaque  champs metric[accuracy] de metrics par le nombre d'exemple de la meme entrée
        for num_examples, metric in metrics   
    ]

    examples = [  # recupère le nombre de données utilisées par chaque clients 
        num_examples
        for num_examples, _ in metrics
    ]

    return {
        "accuracy": sum(accuracies) / sum(examples)  # la moyennen globale 
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

    def evaluate( #fonction appellée automatique
        server_round: int, # numero du round actuel
        parameters: fl.common.NDArrays, #poids du modèle global envoyés par le serveur
        config: Dict[str, fl.common.Scalar], 
    ):
        model = get_model() #crée un nouveau modele tenserflow
        model.set_weights(parameters) #maj des poids pour  le modele global apres agregation

        loss, accuracy = model.evaluate( #evaluation du modele 
            testset,
            verbose=VERBOSE, #controlle de laffichage des détails de levaluation tenserflow
        )

        print(  # l'affichage des réesultats
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

min_partition_size = TOTAL_DATASET_SIZE // NUM_CLIENTS // 10  # la taille minimale des données qu'un client doit recevoir

partitioner = DirichletPartitioner( # créer un système qui découpe le dataset entre les clients
    num_partitions=NUM_CLIENTS, #nombre de clients
    partition_by="label",  #
    alpha=0.5,#controle le niveau de deséquilibre des données entre les clients , plus alpha est petit plus clients tres diffirents
    min_partition_size=min_partition_size,
    self_balancing=True,# evite qu'un client reçoive trop ou peu de données
)

mnist_fds = FederatedDataset( # crée le dataset fédéré MNIST
    dataset="mnist", #dataset utilisé : mnist
    partitioners={"train": partitioner},#applique le découpage fédéré sur les données d’entraînement.
)

centralized_testset = mnist_fds.load_split("test").to_tf_dataset( # charge le datset de test global utilisé par le serveur , et le convertit en format tenserFlow
    columns="image", # les images sont les données d'entrée
    label_cols="label", #les labele sont les bonnes réponses  
    batch_size=BATCH_SIZE, # data set sera envoyé au modele par groupe de(BATCH_SIZE)  images  ici 32
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
    "num_gpus": 1 if tf.config.list_physical_devices("GPU") else 0,
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
