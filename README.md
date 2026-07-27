# Stage-IA-FederatedLearning

Projet réalisé dans le cadre d'un stage sur le **Federated Learning (FL)** : simulation d'entraînement fédéré avec [Flower](https://flower.ai/) (stratégie FedAvg), surveillance des métriques avec détection d'anomalies et alertes email, agent LLM d'analyse de runs (ReAct), et dashboard pédagogique Streamlit.

> **Note :** ce repo ne représente **qu'une partie** du travail réalisé durant le stage, pas le projet complet. Il contient les briques explorées et implémentées au cours de cette partie du travail.

## Contexte du stage

Une grande partie du stage a consisté à **faire de la recherche et comprendre de nouveaux concepts** (comment fonctionne l'apprentissage fédéré, comment fonctionne Flower, etc.) avant de pouvoir les appliquer — cette phase de compréhension a pris du temps et n'est pas toujours visible directement dans le code.

Concrètement :
- Le **code de base** (`fedavg.py`) a d'abord été analysé en profondeur et commenté ligne par ligne pour bien comprendre le fonctionnement de Flower et de FedAvg (modèle, client, partitionnement des données, agrégation).
- Le suivi détaillé des métriques (temps par round, temps par client, etc.) et le système de détection d'événements par seuils avec alertes email (`event_detector.py`) constitutaien une piste **plus compliquée que nécessaire** ; ma tutrice de stage m'a dit de ne pas continuer dans cette direction. Je me suis recentré sur des **métriques plus simples**, celles reprises ensuite par l'agent (accuracy/loss finale, pente d'accuracy, variance, plus grande chute, convergence — voir `compute_signals` dans `fl_agent/tools.py`).
- Après cette phase de compréhension, j'ai construit un **agent ReAct** (`fl_agent/`) pour analyser les runs FedAvg. J'ai d'abord utilisé **Ollama** en local, en testant **plusieurs modèles locaux**, puis j'ai essayé un **LLM externe gratuit (Groq)** pour comparer les résultats entre les différents modèles. 

## Progression du stage

Le projet a été construit progressivement, en partant de la compréhension du code de base jusqu'à l'ajout d'outils d'analyse et d'observabilité :

1. **Simulation FedAvg** (`fedavg.py`) — code de départ analysé et commenté en détail (modèle, client Flower, préparation du dataset fédéré, stratégie d'agrégation).
2. **Métriques détaillées + détection d'événements** (`event_detector.py`) — logging CSV avancé (temps d'évaluation serveur, temps total de round, temps d'entraînement par client, échecs) et détection d'anomalies par seuils avec alertes email. **Piste abandonnée sur conseil de ma tutrice** : trop compliquée par rapport au besoin réel.
3. **Recentrage sur des métriques simples** — accuracy/loss finale, pente d'accuracy, variance, plus grande chute, convergence : ce sont ces signaux, plus simples, qui sont repris dans `fl_agent/tools.py::compute_signals`.
4. **Restructuration du repo** — ajout du `dashboard/` (onboarding Streamlit) et du squelette `fl_agent/`.
5. **Agent ReAct d'analyse** (`fl_agent/`) — construction outil par outil (`load_run`, `compute_signals`, `classify_run`, `write_report`), puis boucle agent (`build_prompt`, `parse_action`, `execute_tool`, fallback anti-blocage), avec itérations successives : plusieurs modèles **Ollama** testés en local (dont `mistral:7b-instruct`), puis intégration de l'API **Groq** (gratuite) pour comparer avec un LLM externe, amélioration de l'anti-blocage et correction d'un bug d'accuracy dans le dashboard.

## Structure du projet

```
.
├── fedavg.py            # Simulation FedAvg (Flower + TensorFlow) sur MNIST
├── event_detector.py    # Surveillance des métriques + alertes email
├── list_models.py       # Liste les modèles Gemini disponibles (utilitaire)
├── environment.yml      # Environnement conda (simulation FL)
├── fl_agent/            # Agent ReAct d'analyse des runs FedAvg
│   ├── agent.py         # Boucle ReAct (prompt, parsing, exécution des outils, fallback)
│   ├── tools.py         # Outils : chargement de run, calcul de signaux, classification, appel LLM, rapport
│   └── run_agent.py     # Point d'entrée CLI
├── dashboard/           # Dashboard Streamlit (onboarding FedAvg)
│   ├── app.py
│   ├── fedavg_runner.py
│   └── runs/            # Exemples de runs (clean / attacks / non-iid)
└── logs/                # Métriques générées par fedavg.py (CSV)
```

## Composants

### 1. Simulation FedAvg (`fedavg.py`)

Simule un entraînement fédéré sur **MNIST** avec [Flower](https://flower.ai/) :
- Partitionnement non-IID des données entre clients (`DirichletPartitioner`)
- Entraînement local par client (TensorFlow/Keras)
- Agrégation FedAvg côté serveur, avec mesure des temps par round et par client
- Écriture des métriques dans `logs/global_metrics.csv` et `logs/client_metrics.csv`

```bash
conda env create -f environment.yml
conda activate fedavg-stage
python fedavg.py
```

### 2. Détection d'événements (`event_detector.py`) — piste explorée, non retenue

Surveille `logs/global_metrics.csv` en continu et détecte des anomalies (convergence, chute d'accuracy, divergence de la loss, client lent, round trop long, échecs de clients). Envoie une alerte par email (Gmail SMTP) à chaque événement détecté.

> Ce module (et le logging CSV détaillé qui va avec, dans `fedavg.py`) a été construit en explorant une approche de suivi/alerting assez poussée, mais ma tutrice de stage m'a orienté vers quelque chose de plus simple. Il reste dans le repo à titre d'exploration ; les métriques réellement reprises par la suite sont celles du point suivant.

Nécessite un fichier `.env` à la racine :

```
GMAIL_USER=...
GMAIL_APP_PASSWORD=...
ALERT_RECIPIENT=...
```

```bash
python event_detector.py
```

### 3. Agent ReAct d'analyse (`fl_agent/`)

Agent qui analyse un run FedAvg (fichier JSON de résultats) en suivant une boucle ReAct : chargement du run, calcul de signaux simples (accuracy/loss finale, pente d'accuracy, variance, plus grande chute, convergence — voir `compute_signals` dans `fl_agent/tools.py`), classification (`clean` / `attacked` / `non_iid`), commentaire généré par LLM, puis génération d'un rapport markdown.

Le choix du LLM est piloté par `fl_agent/tools.py` :
- **Groq** (`ask_groq`, modèle `llama-3.1-8b-instant`) — utilisé par défaut dans la boucle de l'agent
- **Ollama local** (`ask_llm`, modèle `mistral:7b-instruct`, `http://localhost:11434`)

Nécessite les clés correspondantes dans `.env` (`GROQ_API_KEY`, `GEMINI_API_KEY`).

```bash
cd fl_agent
python run_agent.py <dossier_des_runs> --run sample_run_clean
# ou pour analyser tous les runs d'un dossier :
python run_agent.py <dossier_des_runs> --all
```

Les rapports sont écrits dans `fl_agent/reports/`.

### 4. Dashboard Streamlit (`dashboard/`)

Dashboard autonome d'onboarding pour comprendre FedAvg : configuration et lancement d'une expérience, visualisation des courbes accuracy/loss, export des résultats, et un onglet Q&A local (sans clé API) sur les concepts FedAvg.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Prérequis

- Python 3.11 (environnement conda fourni dans `environment.yml`) pour la simulation FL
- Python 3.10+ et les dépendances de `dashboard/requirements.txt` pour le dashboard
- Pour `fl_agent/` : `groq`, `google-genai`, `python-dotenv`, `requests`, `pandas` (pas de fichier `requirements.txt` dédié pour l'instant — à installer manuellement)
- Un fichier `.env` à la racine avec les clés API/SMTP nécessaires selon les modules utilisés

## Logs et données générées

Les dossiers suivants sont générés à l'exécution et ignorés par git (`.gitignore`) : `logs/`, `dashboard/outputs/`, `fl_agent/reports/`.
