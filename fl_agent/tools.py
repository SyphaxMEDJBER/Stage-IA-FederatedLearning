import requests
import json


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
