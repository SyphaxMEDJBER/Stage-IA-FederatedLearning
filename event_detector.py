"""
Détection d'événements pour la plateforme Federated Learning.
Lit global_metrics.csv et détecte les anomalies round par round.
"""

import pandas as pd

# ============================================================
# 1. Chargement des métriques
# ============================================================

def load_metrics():
    """Charge global_metrics.csv et ignore le round 0 (pas de deltas)."""

    df = pd.read_csv("logs/global_metrics.csv") # lire le fichier csv

    # le round 0 est l'évaluation initiale avant tout entraînement,
    # il n'a pas de accuracy_delta ni loss_delta donc on l'ignore
    df = df[df["server_round"] > 0].dropna(subset=["accuracy_delta", "loss_delta"])

    return df


if __name__ == "__main__":
    df = load_metrics()
    print(df)
