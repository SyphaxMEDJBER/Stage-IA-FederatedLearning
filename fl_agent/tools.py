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


def compare_runs(runs):
    """
    compare plusieurs runs entre eux et identifie le meilleur
    prend un dict {nom_run: run_data} et retourne les métriques clés de chaque run
    """
    if len(runs) < 2:
        return {"error": "il faut au moins 2 runs pour comparer"}

    result = {}
    for name, data in runs.items():
        sig = compute_signals(data)       # on calcule les signaux de chaque run
        result[name] = {
            "final_accuracy":    sig["final_accuracy"],
            "max_accuracy":      sig["max_accuracy"],
            "final_loss":        sig["final_loss"],
            "converged":         sig["converged"],
            "max_accuracy_drop": sig["max_accuracy_drop"],
        }

    # on cherche le run avec la meilleure accuracy finale
    best = max(result.items(), key=lambda x: x[1]["final_accuracy"])
    result["__best__"] = best[0]

    return result


def classify_run(signals):
    """
    classifie un run en clean / attacked / non_iid à partir de ses signaux
    fonctionne par vote pondéré : chaque indice ajoute des points à une catégorie
    retourne le label gagnant avec sa confiance et les raisons du choix
    """
    score   = {"clean": 0, "attacked": 0, "non_iid": 0}
    reasons = []

    # indices clean : accuracy haute et modèle convergé
    if signals["final_accuracy"] > 0.88:
        score["clean"] += 2
        reasons.append(f"accuracy finale élevée ({signals['final_accuracy']:.3f})")
    if signals["converged"]:
        score["clean"] += 1
        reasons.append("convergé sur les 3 derniers rounds")

    # indices attacked : chute brutale d'accuracy + rebond de la loss
    if signals["max_accuracy_drop"] < -0.05:
        score["attacked"] += 3
        reasons.append(f"chute d'accuracy de {signals['max_accuracy_drop']:.3f} en un round")
    if signals["num_loss_increases"] >= 2 and signals["max_loss_increase"] > 0.2:
        score["attacked"] += 2
        reasons.append(f"pic de loss détecté ({signals['num_loss_increases']} rounds)")

    # indices non_iid : oscillations fréquentes, convergence lente
    if signals["accuracy_variance"] > 0.001:
        score["non_iid"] += 2
        reasons.append(f"variance élevée de l'accuracy ({signals['accuracy_variance']:.5f})")
    if signals["num_negative_deltas"] >= 2 and signals["max_accuracy_drop"] > -0.05:
        score["non_iid"] += 1
        reasons.append(f"oscillations multiples ({signals['num_negative_deltas']} baisses)")

    # le label avec le plus de points gagne
    label      = max(score, key=score.get)
    confidence = round(score[label] / (sum(score.values()) or 1), 3)

    return {
        "label":      label,
        "confidence": confidence,
        "scores":     score,
        "reasons":    reasons,
    }
