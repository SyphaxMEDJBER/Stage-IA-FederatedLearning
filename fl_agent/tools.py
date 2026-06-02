import requests
import json
import statistics


def ask_llm(prompt, model="phi3:mini", host="http://localhost:11434"):
    """
    envoie un prompt au LLM local (Ollama) et retourne la réponse
    si ollama est pas lancé ça retourne un message d'erreur au lieu de planter
    model par défaut : phi3:mini, host par défaut : localhost:11434
    """
    try:
        resp = requests.post(
            f"{host}/api/generate",          # endpoint HTTP d'Ollama
            json={
                "model": model,
                "prompt": prompt,            # le texte qu'on envoie au LLM
                "stream": False,             # attendre la réponse complète
            },
            timeout=60,                      # abandon si pas de réponse en 60s
        )
        resp.raise_for_status()              # lève une erreur si code HTTP ≠ 200
        return resp.json()["response"]       # extrait le texte de la réponse JSON

    except requests.exceptions.ConnectionError:
        return "[Ollama indisponible — lance : ollama serve]"
    except Exception as e:
        return f"[Erreur LLM : {e}]"


def load_run(path):
    """
    charge un fichier JSON de run FedAvg
    vérifie que les clés obligatoires sont là avant de retourner les données
    lève une ValueError si une clé est manquante
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for key in ("config", "rounds", "global_accuracy", "global_loss"):
        if key not in data:
            raise ValueError(f"Clé manquante '{key}' dans {path}")

    return data


def compute_signals(run_data):
    """
    calcule les signaux diagnostics à partir des courbes accuracy/loss
    retourne un dictionnaire de métriques pour aider à classifier le run
    """
    acc  = run_data["global_accuracy"]
    loss = run_data["global_loss"]
    n    = len(acc)

    # variation d'accuracy et de loss entre chaque round
    acc_deltas  = [acc[i]  - acc[i - 1] for i in range(1, n)]
    loss_deltas = [loss[i] - loss[i - 1] for i in range(1, len(loss))]

    # pente moyenne de l'accuracy sur tous les rounds (régression linéaire simple)
    if n > 1:
        xm    = (n - 1) / 2.0
        ym    = sum(acc) / n
        num   = sum((i - xm) * (acc[i] - ym) for i in range(n))
        den   = sum((i - xm) ** 2 for i in range(n))
        slope = num / den if den else 0.0
    else:
        slope = 0.0

    # convergence : si les 3 derniers rounds varient de moins de 0.5%
    last3     = acc[-3:] if n >= 3 else acc
    converged = (max(last3) - min(last3)) < 0.005

    return {
        "final_accuracy":      acc[-1],
        "max_accuracy":        max(acc),
        "final_loss":          loss[-1],
        "accuracy_slope":      slope,
        "accuracy_variance":   statistics.variance(acc_deltas) if len(acc_deltas) > 1 else 0.0,
        "max_accuracy_drop":   min(acc_deltas) if acc_deltas else 0.0,   # plus grande chute en un round
        "num_negative_deltas": sum(1 for d in acc_deltas if d < 0),      # nb de rounds où l'accuracy a baissé
        "num_loss_increases":  sum(1 for d in loss_deltas if d > 0),     # nb de rounds où la loss a monté
        "max_loss_increase":   max(loss_deltas) if loss_deltas else 0.0,
        "converged":           converged,
        "is_monotonic":        all(d >= 0 for d in acc_deltas),          # accuracy toujours croissante
    }
