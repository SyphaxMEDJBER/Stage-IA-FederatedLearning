"""
Détection d'événements pour la plateforme Federated Learning.
Lit global_metrics.csv et détecte les anomalies round par round.
"""

import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # charge les variables depuis le fichier .env

# identifiants email chargés depuis .env
GMAIL_USER       = os.getenv("GMAIL_USER")
GMAIL_PASSWORD   = os.getenv("GMAIL_APP_PASSWORD")
ALERT_RECIPIENT  = os.getenv("ALERT_RECIPIENT")

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
# 2. Envoi des alertes par email
# ============================================================

MESSAGES = {
    "normal":         lambda m: f"Round normal — accuracy_delta={m['accuracy_delta']:.4f}, loss_delta={m['loss_delta']:.4f}",
    "convergence":    lambda m: f"Le modèle a convergé — accuracy_delta={m['accuracy_delta']:.4f}",
    "accuracy_drop":  lambda m: f"Chute d'accuracy de {m['accuracy_delta']:.4f} !",
    "divergence":     lambda m: f"La loss monte ! loss_delta={m['loss_delta']:.4f}",
    "slow_client":    lambda m: f"Client lent — max={m['max_client_training_time']:.1f}s vs avg={m['avg_client_training_time']:.1f}s",
    "round_too_long": lambda m: f"Round trop long — {m['full_round_time']:.1f}s",
    "low_accuracy":   lambda m: f"Accuracy trop basse — {m['accuracy']:.4f}",
    "client_failure": lambda m: f"{int(m['num_failures'])} client(s) ont planté !",
}

def send_alert(event, server_round, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail    = MESSAGES.get(event, lambda _: "")(metrics)
    message   = f"[{timestamp}] [{event.upper()}] Round {server_round} — {detail}"

    print(message)

    try:
        sujet = f"[FL Alert] {event.upper()} — Round {server_round}"
        corps = f"{message}\n\nMétriques complètes :\n{metrics}"

        msg            = MIMEText(corps)
        msg["Subject"] = sujet
        msg["From"]    = GMAIL_USER
        msg["To"]      = ALERT_RECIPIENT

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)

    except Exception as e:
        print(f"Erreur envoi email : {e}")


# ============================================================
# 3. Détection des événements
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


# ============================================================
# 4. File watcher — surveillance du CSV en temps réel
# ============================================================

def watch_metrics(interval=3):
    """
    Surveille global_metrics.csv en boucle et détecte les événements
    dè s qu'un nouveauround apparaît dans le fichier.
    interval : délai en secondes entre chaque lecture du CSV
    """
    last_round = 0  # numéro du dernier round déjà traité

    print("Surveillance démarrée... (Ctrl+C pour arrêter)")

    while True:
        df = load_metrics()

        # garder uniquement les rounds pas encore traités
        new_rows = df[df["server_round"] > last_round]

        for _, row in new_rows.iterrows():
            metrics = row.to_dict()  # convertir la ligne CSV en dictionnaire
            events  = detect_events(metrics)  # détecter les événements

            for event in events:
                send_alert(event, int(metrics["server_round"]), metrics)

            last_round = int(metrics["server_round"])  # mettre à jour le dernier round traité

        time.sleep(interval)  # attendre avant de relire le CSV


if __name__ == "__main__":
    watch_metrics()
