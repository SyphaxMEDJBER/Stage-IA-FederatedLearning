import os
import sys
import argparse

from agent import run_agent


def main():
    parser = argparse.ArgumentParser(description="FedAvg ReAct agent — analyse des runs avec Ollama")

    # argument obligatoire : le dossier contenant les fichiers JSON
    parser.add_argument("runs_dir", help="chemin vers le dossier des runs JSON")

    # options
    parser.add_argument("--run",  default=None, help="nom du run à analyser (sans .json)")
    parser.add_argument("--all",  action="store_true", help="analyser tous les fichiers JSON du dossier")

    args = parser.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    if not os.path.isdir(runs_dir):
        print(f"Erreur : '{runs_dir}' n'est pas un dossier valide.")
        sys.exit(1)

    # détermine les runs à analyser
    if args.all:
        run_names = [os.path.splitext(f)[0] for f in os.listdir(runs_dir) if f.endswith(".json")]
    elif args.run:
        run_names = [args.run]
    else:
        run_names = ["sample_run_clean"]   # run par défaut

    print("=" * 60)
    print("FedAvg ReAct Agent")
    print(f"Répertoire : {runs_dir}")
    print(f"Runs à analyser : {run_names}")
    print("=" * 60)

    for run_name in run_names:
        print(f"\n>>> Analyse de : {run_name}")
        print("-" * 40)
        state = run_agent(runs_dir=runs_dir, target_run=run_name)

        if state["report_path"]:
            print(f"\nRapport généré : {state['report_path']}")
        else:
            print("Aucun rapport généré.")


if __name__ == "__main__":
    main()
