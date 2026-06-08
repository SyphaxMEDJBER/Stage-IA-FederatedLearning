import os
import json
import statistics
import requests
from dotenv import load_dotenv
from google import genai as google_genai
from groq import Groq

load_dotenv()

def ask_groq(prompt, model="llama-3.1-8b-instant"):
    """
    envoie un prompt à l'API Groq et retourne la réponse
    utilise la clé GROQ_API_KEY du fichier .env
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "[GROQ_API_KEY manquante dans .env]"
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Erreur Groq : {e}]"


def ask_gemini(prompt, model="gemini-2.0-flash-lite"):
    """
    envoie un prompt à l'API Gemini et retourne la réponse
    utilise la clé GEMINI_API_KEY du fichier .env
    """
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "[GEMINI_API_KEY manquante dans .env]"
        client   = google_genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text
    except Exception as e:
        return f"[Erreur Gemini : {e}]"


def ask_llm(prompt, model="mistral:7b-instruct", host="http://localhost:11434"):
    """
    envoie un prompt au LLM local (Ollama) et retourne la réponse
    si ollama est pas lancé ça retourne un message d'erreur au lieu de planter
    model par défaut : phi3:mini, host par défaut : localhost:11434
    """
    try:
        resp = requests.post(     # requette http
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
    with open(path, encoding="utf-8") as f: #on ouvre le fichier en lecture , avec with il se fermera auto
        data = json.load(f) #lit le fichier et le convertit en dictionnaire python

    for key in ("config", "rounds", "global_accuracy", "global_loss"):# on verifie que les 4 clés essentielles sont presentes 
        if key not in data:# si une manque on lève une exception 
            raise ValueError(f"Clé manquante '{key}' dans {path}")

    return data


def compute_signals(run_data):
    """
    prends le dictionnaire retourné par load_run
    calcule les signaux diagnostics à partir des courbes accuracy/loss
    retourne un dictionnaire de métriques pour aider à classifier le run
    """
    acc  = run_data["global_accuracy"] # liste des accuracy 
    loss = run_data["global_loss"] # liste des loss
    n    = len(acc) # nombre de rounds

    # variation d'accuracy et de loss entre chaque round
    acc_deltas  = [acc[i]  - acc[i - 1] for i in range(1, n)]
    loss_deltas = [loss[i] - loss[i - 1] for i in range(1, len(loss))]

    # pente moyenne de l'accuracy sur tous les rounds (régression linéaire simple)
    if n > 1: # il faut au moins 2 points pour tracer une droite 
        xm    = (n - 1) / 2.0 # la moyenne des positions (numeros des rounds)
        ym    = sum(acc) / n # la moyenne des accuracy
        num   = sum((i - xm) * (acc[i] - ym) for i in range(n))# pour chaque round on multiplie l'ecart du round parraport zu centre par l'eccart du laccuracy paraport a la moyenne   
        den   = sum((i - xm) ** 2 for i in range(n))# normaliser le resultat
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
    # seuil à 0.003 : en dessous c'est la variance normale d'un run clean qui monte régulièrement
    if signals["accuracy_variance"] > 0.003:
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


def write_report(run_name, run_data, signals, classification, llm_commentary, output_path):
    """
    génère un rapport markdown pour un run analysé
    rassemble la config, les signaux, la classification et le commentaire du LLM
    retourne le chemin du fichier créé
    """
    cfg  = run_data.get("config", {})

    # on construit le rapport ligne par ligne sous forme de liste
    lines = [
        f"# Rapport d'analyse — {run_name}",
        "",
        "## Configuration",
        "",
        "| Paramètre | Valeur |",
        "|---|---|",
    ]
    for k, v in cfg.items():
        lines.append(f"| `{k}` | {v} |")   # une ligne par paramètre du run

    # section classification : résultat et raisons du choix
    lines += [
        "",
        "## Classification",
        "",
        f"**Résultat : `{classification['label'].upper()}`**  ",
        f"**Confiance : {classification['confidence']:.0%}**",
        "",
        "**Indices détectés :**",
    ]
    for r in classification["reasons"]:
        lines.append(f"- {r}")

    # section signaux : les métriques calculées par compute_signals
    lines += [
        "",
        "## Signaux clés",
        "",
        "| Signal | Valeur |",
        "|---|---|",
        f"| accuracy finale | {signals['final_accuracy']:.4f} |",
        f"| accuracy max | {signals['max_accuracy']:.4f} |",
        f"| loss finale | {signals['final_loss']:.4f} |",
        f"| convergé | {signals['converged']} |",
        f"| pente accuracy | {signals['accuracy_slope']:.5f} |",
        f"| plus grande chute | {signals['max_accuracy_drop']:.4f} |",
        f"| rounds en baisse | {signals['num_negative_deltas']} |",
        f"| pics de loss | {signals['num_loss_increases']} |",
        "",
        "## Commentaire LLM",
        "",
        llm_commentary,   # texte retourné par ask_llm
        "",
        "---",
        "*rapport généré par le ReAct agent*",
    ]

    # crée le dossier si nécessaire et écrit le fichier markdown
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
