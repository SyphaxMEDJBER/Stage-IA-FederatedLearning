import os
import json
import re

from tools import load_run, compute_signals, compare_runs, classify_run, ask_llm, write_report

MAX_ITERATIONS = 10


# ============================================================
# 1. Construction du prompt envoyé à Ollama à chaque itération
# ============================================================

def build_prompt(state):
    # résume tout ce qui a été fait jusqu'ici pour qu'Ollama sache où on en est
    history_lines = []
    for step in state["history"]:
        history_lines.append(f"[Étape {step['step']}]")
        history_lines.append(f"Pensée : {step['thought']}")
        history_lines.append(f"Action : {step['action']['name']}({step['action'].get('args', {})})")
        history_lines.append(f"Résultat : {step['observation']}")

    history_str = "\n".join(history_lines) if history_lines else "aucune étape encore."

    loaded     = list(state["loaded_runs"].keys())
    classified = {k: v["label"] for k, v in state["classifications"].items()}

    # on injecte l'état courant + les outils disponibles dans le prompt
    return f"""Tu es un agent d'analyse de runs FedAvg.

Répertoire analysé : {state["runs_dir"]}
Fichiers disponibles : {state["available_files"]}
Runs chargés : {loaded}
Signaux calculés pour : {list(state["signals"].keys())}
Classifications : {classified}
Rapport écrit : {state["report_path"] is not None}

Historique :
{history_str}

Outils disponibles :
1. load_run(path)                      - charge un fichier JSON de run
2. compute_signals(run_name)           - calcule les signaux d'un run chargé
3. compare_runs(run_names)             - compare plusieurs runs chargés
4. classify_run(run_name)              - classifie un run (clean/attacked/non_iid)
5. ask_llm(question, context)          - pose une question au LLM
6. write_report(run_name, output_path) - génère le rapport markdown
7. finish()                            - termine l'analyse

Réponds UNIQUEMENT en JSON valide, sans texte autour :
{{
  "thought": "ce que tu penses faire et pourquoi",
  "action": {{
    "name": "nom_de_loutil",
    "args": {{...}}
  }}
}}
"""


# ============================================================
# 2. Extraction de l'action depuis la réponse d'Ollama
# ============================================================

def parse_action(response):
    # ollama doit retourner du JSON, mais parfois il ajoute du texte autour
    # on essaie d'abord un parsing direct
    try:
        data = json.loads(response.strip())
        if "action" in data and "name" in data["action"]:
            return data
    except json.JSONDecodeError:
        pass

    # si ça échoue, on cherche le bloc JSON dans le texte
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "action" in data and "name" in data["action"]:
                return data
        except json.JSONDecodeError:
            pass

    return None  # réponse malformée, le fallback prendra le relais
