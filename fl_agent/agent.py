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
    target     = state["target_run"]

    # on calcule explicitement ce qui reste à faire pour guider Ollama
    todo = []
    if target not in state["loaded_runs"]:
        todo.append(f"1. load_run(path='{target}.json')")
    if target not in state["signals"]:
        todo.append(f"2. compute_signals(run_name='{target}')")
    if target not in state["classifications"]:
        todo.append(f"3. classify_run(run_name='{target}')")
    if not state["report_path"]:
        todo.append(f"4. write_report(run_name='{target}', output_path='')")
    if not todo:
        todo.append("5. finish()")

    todo_str = "\n".join(todo)

    # on injecte l'état courant + les outils disponibles dans le prompt
    return f"""Tu es un agent d'analyse de runs FedAvg. Run cible : '{target}'

État actuel :
- Runs chargés : {loaded}
- Signaux calculés : {list(state["signals"].keys())}
- Classifications : {classified}
- Rapport écrit : {state["report_path"] is not None}

Prochaines étapes OBLIGATOIRES dans cet ordre :
{todo_str}

Historique récent :
{history_str}

Outils disponibles :
1. load_run(path)                      - charge un fichier JSON de run
2. compute_signals(run_name)           - calcule les signaux d'un run chargé
3. compare_runs(run_names)             - compare plusieurs runs chargés
4. classify_run(run_name)              - classifie un run (clean/attacked/non_iid)
5. ask_llm(question, context)          - pose une question au LLM
6. write_report(run_name, output_path) - génère le rapport markdown
7. finish()                            - termine l'analyse

Fais UNIQUEMENT la première étape de la liste ci-dessus.
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

    # ollama renvoie parfois args comme une liste au lieu d'un dict → on ignore
    if not isinstance(args, dict):
        args = {}

    try:
        if name == "load_run":
            path = args.get("path", "")
            # si path est vide (args malformés), on cherche le run cible dans les fichiers disponibles
            if not path:
                target   = state["target_run"]
                matching = [f for f in state["available_files"] if target in f]
                path     = matching[0] if matching else (state["available_files"][0] if state["available_files"] else "")
            if not path:
                return "erreur : aucun fichier disponible à charger"
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
            # si run_name est vide ou malformé, on prend le run cible par défaut
            if not run_name or run_name not in state["loaded_runs"]:
                run_name = state["target_run"]
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
            # si run_names est vide ou malformé, on compare tous les runs chargés
            if not isinstance(run_names, list) or not run_names:
                run_names = list(state["loaded_runs"].keys())
            missing = [n for n in run_names if n not in state["loaded_runs"]]
            if missing:
                return f"erreur : runs pas chargés : {missing}"
            subset = {n: state["loaded_runs"][n] for n in run_names}
            result = compare_runs(subset)
            state["comparisons"] = result
            return f"comparaison terminée — meilleur run : {result.get('__best__')}"

        elif name == "classify_run":
            run_name = args.get("run_name", "")
            # si run_name est vide ou malformé, on prend le run cible par défaut
            if not run_name or run_name not in state["signals"]:
                run_name = state["target_run"]
            if run_name not in state["signals"]:
                return f"erreur : signaux manquants pour '{run_name}'"
            result = classify_run(state["signals"][run_name])
            state["classifications"][run_name] = result          # on stocke la classification
            return (f"classification '{run_name}' : {result['label'].upper()} "
                    f"(confiance={result['confidence']:.0%})")

        elif name == "ask_llm":
            question = args.get("question", "")
            context  = args.get("context", "")
            # on force Ollama à rester sur le sujet FedAvg avec un prompt structuré
            forced_prompt = (
                f"Tu es un expert en Federated Learning. Réponds uniquement sur le run FedAvg analysé.\n"
                f"Contexte du run : {context}\n"
                f"Question : {question}\n"
                f"Réponse courte (3-5 phrases max), en français, uniquement sur ce run :"
            )
            response = ask_llm(forced_prompt)
            state["llm_responses"].append({"question": question, "response": response})
            return f"réponse LLM : {response[:200]}"

        elif name == "write_report":
            run_name    = args.get("run_name", "")
            output_path = args.get("output_path", "")
            # si run_name est vide ou malformé, on prend le run cible par défaut
            if not run_name or run_name not in state["loaded_runs"]:
                run_name = state["target_run"]
            if not output_path:
                output_path = os.path.join(state["runs_dir"], "..", "..", "fl_agent", "reports", f"{run_name}_analysis.md")
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


# ============================================================
# 5. Boucle principale ReAct
# ============================================================

def run_agent(runs_dir, target_run="sample_run_clean"):
    available = [f for f in os.listdir(runs_dir) if f.endswith(".json")] if os.path.isdir(runs_dir) else []

    # état partagé entre toutes les itérations
    state = {
        "runs_dir":        runs_dir,
        "available_files": available,
        "target_run":      target_run,
        "loaded_runs":     {},
        "signals":         {},
        "classifications": {},
        "comparisons":     {},
        "llm_responses":   [],
        "report_path":     None,
        "finished":        False,
        "history":         [],
    }

    print(f"\n[Agent] démarrage — répertoire : {runs_dir}")
    print(f"[Agent] fichiers disponibles : {available}")

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Agent] itération {iteration + 1}/{MAX_ITERATIONS}")

        # on s'arrête si le rapport est écrit ou si l'agent a appelé finish()
        if state["finished"] or state["report_path"]:
            print("[Agent] tâche terminée.")
            break

        # on envoie l'état courant à Ollama pour décider la prochaine action
        prompt   = build_prompt(state)
        response = ask_llm(prompt)
        print(f"[Agent] réponse Ollama (extrait) : {response[:150]}")

        parsed = parse_action(response)

        # si Ollama retourne du JSON malformé, on utilise le fallback
        if parsed is None:
            print("[Agent] réponse malformée — utilisation du fallback")
            fallback = get_fallback_action(state)
            if fallback is None:
                print("[Agent] aucune action fallback disponible, arrêt.")
                break
            parsed = {"thought": "fallback automatique", "action": fallback}

        # détection de blocage : même action 3 fois de suite → on force l'étape suivante
        recent = [s["action"]["name"] for s in state["history"][-3:]]
        if len(recent) == 3 and len(set(recent)) == 1 and recent[0] == parsed["action"]["name"]:
            print(f"[Agent] blocage sur '{parsed['action']['name']}' — forçage du fallback")
            fallback = get_fallback_action(state)
            if fallback:
                parsed = {"thought": "anti-blocage", "action": fallback}

        thought = parsed.get("thought", "")
        action  = parsed["action"]

        observation = execute_tool(parsed, state)

        print(f"[Agent] pensée   : {thought[:100]}")
        print(f"[Agent] action   : {action['name']}")
        print(f"[Agent] résultat : {observation[:150]}")

        # on sauvegarde cette itération dans l'historique
        state["history"].append({
            "step":        iteration + 1,
            "thought":     thought,
            "action":      action,
            "observation": observation,
        })

    # si on a épuisé les 10 itérations sans rapport, on en génère un quand même
    if not state["report_path"]:
        print("[Agent] max itérations atteint — génération du rapport de secours")
        _fallback_report(state)

    return state


# ============================================================
# 6. Rapport de secours si max_iterations atteint
# ============================================================

def _fallback_report(state):
    # génère un rapport minimal avec ce qu'on a déjà calculé
    for run_name, run_data in state["loaded_runs"].items():
        if run_name not in state["signals"]:
            state["signals"][run_name] = compute_signals(run_data)
        if run_name not in state["classifications"]:
            state["classifications"][run_name] = classify_run(state["signals"][run_name])

        output_path = os.path.join(state["runs_dir"], "..", "..", "fl_agent", "reports", f"{run_name}_analysis.md")
        path = write_report(
            run_name       = run_name,
            run_data       = run_data,
            signals        = state["signals"][run_name],
            classification = state["classifications"][run_name],
            llm_commentary = "[rapport de secours — max itérations atteint]",
            output_path    = output_path,
        )
        state["report_path"] = path
        print(f"[Agent] rapport de secours écrit : {path}")
        break
