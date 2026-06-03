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


# ============================================================
# 3. Exécution d'un outil et mise à jour de l'état
# ============================================================

def execute_tool(parsed, state):
    name = parsed["action"]["name"]
    args = parsed["action"].get("args", {})

    try:
        if name == "load_run":
            path = args.get("path", "")
            if not os.path.isabs(path):                          # si chemin relatif, on le complète
                path = os.path.join(state["runs_dir"], path)
            data     = load_run(path)
            run_name = os.path.splitext(os.path.basename(path))[0]
            state["loaded_runs"][run_name] = data                # on stocke le run dans l'état
            cfg  = data.get("config", {})
            summ = data.get("summary", {})
            return (f"run '{run_name}' chargé : {cfg.get('num_clients')} clients, "
                    f"{cfg.get('num_rounds')} rounds, "
                    f"accuracy finale={summ.get('final_accuracy')}, "
                    f"convergé={summ.get('converged')}")

        elif name == "compute_signals":
            run_name = args.get("run_name", "")
            if run_name not in state["loaded_runs"]:
                return f"erreur : run '{run_name}' pas encore chargé"
            sig = compute_signals(state["loaded_runs"][run_name])
            state["signals"][run_name] = sig                     # on stocke les signaux dans l'état
            return (f"signaux calculés pour '{run_name}' : "
                    f"slope={sig['accuracy_slope']:.4f}, "
                    f"max_drop={sig['max_accuracy_drop']:.4f}, "
                    f"convergé={sig['converged']}")

        elif name == "compare_runs":
            run_names = args.get("run_names", [])
            missing   = [n for n in run_names if n not in state["loaded_runs"]]
            if missing:
                return f"erreur : runs pas chargés : {missing}"
            subset = {n: state["loaded_runs"][n] for n in run_names}
            result = compare_runs(subset)
            state["comparisons"] = result
            return f"comparaison terminée — meilleur run : {result.get('__best__')}"

        elif name == "classify_run":
            run_name = args.get("run_name", "")
            if run_name not in state["signals"]:
                return f"erreur : signaux manquants pour '{run_name}'"
            result = classify_run(state["signals"][run_name])
            state["classifications"][run_name] = result          # on stocke la classification
            return (f"classification '{run_name}' : {result['label'].upper()} "
                    f"(confiance={result['confidence']:.0%})")

        elif name == "ask_llm":
            question = args.get("question", "")
            context  = args.get("context", "")
            response = ask_llm(f"Contexte : {context}\n\nQuestion : {question}")
            state["llm_responses"].append({"question": question, "response": response})
            return f"réponse LLM : {response[:200]}"

        elif name == "write_report":
            run_name    = args.get("run_name", "")
            output_path = args.get("output_path", "")
            if not output_path:
                output_path = os.path.join(state["runs_dir"], "..", "fl_agent", "reports", f"{run_name}_analysis.md")
            if run_name not in state["loaded_runs"]:
                return f"erreur : run '{run_name}' pas chargé"
            if run_name not in state["signals"]:
                return f"erreur : signaux manquants pour '{run_name}'"
            if run_name not in state["classifications"]:
                return f"erreur : classification manquante pour '{run_name}'"
            commentary = "\n".join(r["response"] for r in state["llm_responses"]) or "aucun commentaire LLM"
            path = write_report(
                run_name       = run_name,
                run_data       = state["loaded_runs"][run_name],
                signals        = state["signals"][run_name],
                classification = state["classifications"][run_name],
                llm_commentary = commentary,
                output_path    = output_path,
            )
            state["report_path"] = path
            return f"rapport écrit : {path}"

        elif name == "finish":
            state["finished"] = True
            return "analyse terminée"

        else:
            return f"outil inconnu : '{name}'"

    except Exception as e:
        return f"erreur lors de '{name}' : {type(e).__name__} : {e}"


# ============================================================
# 4. Fallback : action logique suivante si Ollama échoue
# ============================================================

def get_fallback_action(state):
    # si Ollama retourne du JSON malformé, on décide nous-mêmes la prochaine étape
    target = state["target_run"]

    if target not in state["loaded_runs"]:
        matching = [f for f in state["available_files"] if target in f]
        path = matching[0] if matching else (state["available_files"][0] if state["available_files"] else None)
        if path:
            return {"name": "load_run", "args": {"path": path}}

    if target not in state["signals"]:
        return {"name": "compute_signals", "args": {"run_name": target}}

    if target not in state["classifications"]:
        return {"name": "classify_run", "args": {"run_name": target}}

    if state["report_path"] is None:
        return {"name": "write_report", "args": {"run_name": target, "output_path": ""}}

    return {"name": "finish", "args": {}}
