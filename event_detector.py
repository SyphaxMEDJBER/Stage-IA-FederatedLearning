"""
Détection d'événements pour la plateforme Federated Learning.
Lit global_metrics.csv et détecte les anomalies round par round.
"""

import pandas as pd

# ============================================================
# 1. Seuils de détection
# ============================================================

# chaque seuil définit la limite à partir de laquelle un événement est déclenché
# ces valeurs sont cohérentes avec les ranges de agent_training_data.csv

SEUILS = {
    # --- détection sur les deltas ---
    "convergence_acc_delta_max":  0.01,  # |accuracy_delta| < 0.01  → le modèle a convergé
    "convergence_loss_delta_max": 0.15,  # |loss_delta| < 0.15      → le modèle a convergé
    "accuracy_drop_min":         -0.05,  # accuracy_delta < -0.05   → chute d'accuracy
    "divergence_loss_min":        0.50,  # loss_delta > 0.5         → la loss monte

    # --- détection sur les temps ---
    "slow_client_ratio":          2.50,  # max/avg > 2.5            → un client est anormalement lent
    "round_time_max":           300.0,   # full_round_time > 300s   → round trop long (problème global)

    # --- détection sur les valeurs absolues ---
    "accuracy_min":               0.30,  # accuracy < 0.30          → modèle peu performant
    "failure_min":                1,     # num_failures >= 1        → des clients ont planté
}


# ============================================================
# 2. Détection des événements
# ============================================================

def detect_events(metrics):
    """
    Reçoit un dictionnaire de métriques d'un round et retourne la liste des événements détectés.
    Un même round peut avoir plusieurs événements simultanément.
    """
    events = []  # liste des événements détectés pour ce round

    # convergence : accuracy et loss ne bougent presque plus
    if (abs(metrics["accuracy_delta"]) < SEUILS["convergence_acc_delta_max"] and
            abs(metrics["loss_delta"])    < SEUILS["convergence_loss_delta_max"]):
        events.append("convergence")

    # chute d'accuracy : l'accuracy a baissé de façon significative
    if metrics["accuracy_delta"] < SEUILS["accuracy_drop_min"]:
        events.append("accuracy_drop")

    # divergence : la loss augmente au lieu de descendre
    if metrics["loss_delta"] > SEUILS["divergence_loss_min"]:
        events.append("divergence")

    # client lent : max_training_time très supérieur à avg_training_time
    if (metrics["avg_client_training_time"] > 0 and
            metrics["max_client_training_time"] / metrics["avg_client_training_time"] > SEUILS["slow_client_ratio"]):
        events.append("slow_client")

    # round trop long : full_round_time peut être None (round 0), on vérifie d'abord qu'il existe
    if metrics["full_round_time"] and metrics["full_round_time"] > SEUILS["round_time_max"]:
        events.append("round_too_long")# si le temps de round existe et il est supérieur au seuil

    # accuracy trop basse : le modèle ne performe pas suffisamment
    if metrics["accuracy"] < SEUILS["accuracy_min"]:
        events.append("low_accuracy")

    # failure : des clients ont planté pendant le round
    if metrics["num_failures"] >= SEUILS["failure_min"]:
        events.append("client_failure")

    # si aucun événement anormal détecté → round normal
    if not events:
        events.append("normal")

    return events  # ex: ["slow_client", "accuracy_drop"] ou ["normal"]


# ============================================================
# 3. Chargement des métriques
# ============================================================

def load_metrics():
    """Charge global_metrics.csv et ignore le round 0 (pas de deltas)."""

    df = pd.read_csv("logs/global_metrics.csv") # lire le fichier csv

    # le round 0 est l'évaluation initiale avant tout entraînement,
    # il n'a pas de accuracy_delta ni loss_delta donc on l'ignore
    df = df[df["server_round"] > 0].dropna(subset=["accuracy_delta", "loss_delta"])

    return df


if __name__ == "__main__":#execute le fichier que quand on le lance  , pas quand un autre fichier l'importe 
    df = load_metrics()
    print(df)
