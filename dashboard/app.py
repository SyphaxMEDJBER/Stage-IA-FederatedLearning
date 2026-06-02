"""
app.py — FLows FedAvg Intern Dashboard
=======================================
Simplified Flower/FedAvg dashboard for L3 internship onboarding.

Run:
    streamlit run app.py

Tabs:
  1. Overview        — what FedAvg is, how it works
  2. Run Experiment  — configure and launch a FedAvg run
  3. Results         — global accuracy / loss metrics
  4. Curves          — per-round accuracy and loss charts
  5. Export          — download raw JSON results
  6. FedAvg Notes    — local Q&A helper (no API key required)
"""

import json
import os
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
RUNS_DIR = os.path.join(OUTPUTS_DIR, "runs")
LATEST_JSON = os.path.join(OUTPUTS_DIR, "latest.json")

os.makedirs(RUNS_DIR, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedAvg Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    [data-testid="stSidebar"] { background: #1c2b3a; }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stMetric"] { background: #f0f4fa; border-radius: 8px; padding: 8px; }
    h1 { color: #1c2b3a; border-bottom: 2px solid #1565C0; padding-bottom: 6px; }
    h2 { color: #1a3a5c; }
    h3 { color: #2c4a6a; }
    code { background: #eef2f7; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_latest_results() -> Optional[dict]:
    if not os.path.exists(LATEST_JSON):
        return None
    try:
        with open(LATEST_JSON) as f:
            ref = json.load(f)
        path = ref.get("path", "")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_results_from_path(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def list_past_runs() -> list:
    if not os.path.exists(RUNS_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(RUNS_DIR) if f.endswith("_results.json")],
        reverse=True,
    )
    return [(f, os.path.join(RUNS_DIR, f)) for f in files]


def plotly_accuracy(results: dict) -> go.Figure:
    rounds = results["rounds"]
    accuracy = [a * 100 for a in results["global_accuracy"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=accuracy,
        mode="lines+markers",
        name="FedAvg — global model",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        xaxis_title="Communication Round",
        yaxis_title="Test Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        title="Global Model Accuracy per Round",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def plotly_loss(results: dict) -> go.Figure:
    rounds = results["rounds"]
    losses = results["global_loss"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=losses,
        mode="lines+markers",
        name="FedAvg — global model",
        line=dict(color="#C62828", width=2),
        marker=dict(size=6, symbol="square"),
    ))
    fig.update_layout(
        xaxis_title="Communication Round",
        yaxis_title="Cross-Entropy Loss",
        title="Global Model Loss per Round",
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌐 FedAvg Dashboard")
    st.caption("FedAvg Onboarding")
    st.divider()

    st.subheader("Experiment Configuration")

    st.selectbox("Dataset", ["MNIST"], index=0, disabled=True)
    st.selectbox("Strategy", ["FedAvg (standard)"], index=0, disabled=True)

    num_clients = st.slider("Number of clients", min_value=2, max_value=20, value=4, step=1)
    num_rounds = st.slider("Number of rounds", min_value=2, max_value=30, value=5, step=1)
    fraction_fit = st.slider(
        "fraction_fit", min_value=0.1, max_value=1.0, value=1.0, step=0.1,
        help="Fraction of clients selected for training each round."
    )
    fraction_evaluate = st.slider(
        "fraction_evaluate", min_value=0.1, max_value=1.0, value=1.0, step=0.1,
        help="Fraction of clients used for evaluation each round."
    )
    local_epochs = st.slider("Local epochs", min_value=1, max_value=10, value=1, step=1,
                             help="Number of local training epochs per client per round.")
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    st.divider()
    n_selected = max(1, int(num_clients * fraction_fit))
    st.caption(
        f"→ {n_selected} / {num_clients} clients train each round  \n"
        f"→ {local_epochs} local epoch(s), batch size {batch_size}  \n"
        f"→ Dataset: MNIST (60k train / 10k test)"
    )

    if num_clients > 8 or num_rounds > 20:
        st.warning(
            f"⚠ Large config: {num_clients} clients × {num_rounds} rounds may be slow.\n\n"
            "Recommended for quick experiments: ≤8 clients, ≤10 rounds."
        )

    st.divider()
    run_btn = st.button("▶  Run Experiment", type="primary", use_container_width=True)
    st.divider()
    st.caption("No external API key required.")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📖 Overview",
    "🚀 Run Experiment",
    "📊 Results",
    "📈 Curves",
    "💾 Export",
    "📝 FedAvg Notes",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.title("FedAvg Dashboard")
    st.markdown("**this is an onboarding session**")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### What is Federated Learning?

        Federated Learning (FL) is a machine learning approach where a model is trained
        across multiple decentralized clients (e.g., devices or organizations) **without
        sharing raw data**. Each client keeps its data local and only shares model updates
        (gradients or weights) with a central server.

        ---

        ### How FedAvg works — one round

        1. **The server sends the global model** to a selected set of clients.
        2. **Each client trains locally** on its own data for a few epochs.
        3. **Clients send their updated model weights** back to the server.
        4. **The server averages the weights** (weighted by the number of training samples).
        5. The averaged result becomes the **new global model** for the next round.
        6. Data **never leaves** the client devices.

        ---

        ### Key parameters

        | Parameter | Description |
        |---|---|
        | `num_clients` | Total number of federated clients |
        | `num_rounds` | Number of communication rounds |
        | `fraction_fit` | Fraction of clients selected for training each round |
        | `fraction_evaluate` | Fraction of clients used to evaluate the global model |
        | `local_epochs` | How many local training epochs each client runs |
        | `batch_size` | Mini-batch size for local training |

        ---

        ### What this dashboard does

        - Runs a standard FedAvg experiment on **MNIST** (handwritten digit recognition).
        - The data is split evenly across clients (IID partition).
        - You can observe how accuracy and loss evolve across rounds.
        - Results can be exported as JSON for further analysis.

        """)

    with col2:
        st.markdown("### Quick Start")
        st.code(
            "# Install dependencies\n"
            "pip install -r requirements.txt\n\n"
            "# Launch the dashboard\n"
            "streamlit run app.py",
            language="bash"
        )

        st.markdown("### Suggested first experiment")
        st.info(
            "Clients: 4  \nRounds: 5  \nfraction_fit: 1.0  \n"
            "Local epochs: 1  \nBatch size: 32  \nSeed: 42  \n\n"
            "Expected runtime: ~2–5 min\n\n"
            "Set these in the sidebar and click **▶ Run Experiment**."
        )

        st.markdown("### FedAvg reference")
        st.markdown(
            "McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks "
            "from Decentralized Data*. The original FedAvg paper."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RUN EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Run FedAvg Experiment")

    with st.expander("Current configuration", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Clients", num_clients)
        c2.metric("Rounds", num_rounds)
        c3.metric("fraction_fit", f"{fraction_fit:.1f}")
        c4.metric("fraction_eval", f"{fraction_evaluate:.1f}")
        c5.metric("Local epochs", local_epochs)
        c6.metric("Batch size", batch_size)

    if run_btn:
        st.info("Launching FedAvg simulation — this may take a few minutes.")
        log_area = st.empty()
        progress_bar = st.progress(0)

        runner_script = os.path.join(ROOT, "fedavg_runner.py")

        cmd = [
            sys.executable, runner_script,
            "--clients", str(num_clients),
            "--rounds", str(num_rounds),
            "--fraction_fit", str(fraction_fit),
            "--fraction_evaluate", str(fraction_evaluate),
            "--local_epochs", str(local_epochs),
            "--batch_size", str(batch_size),
            "--seed", str(seed),
        ]

        log_lines = []
        rounds_seen = set()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=ROOT,
            )

            for line in iter(process.stdout.readline, ""):
                line = line.rstrip()
                log_lines.append(line)
                log_area.code("\n".join(log_lines[-35:]))

                if "[Round " in line:
                    for tok in line.split():
                        tok_clean = tok.strip("[]/")
                        if "/" in tok_clean:
                            try:
                                rnd, total = tok_clean.split("/")
                                rounds_seen.add(int(rnd))
                                pct = min(int(len(rounds_seen) / max(int(total), 1) * 100), 99)
                                progress_bar.progress(pct)
                            except ValueError:
                                pass
                            break

            process.wait()
            progress_bar.progress(100)

            if process.returncode == 0:
                st.success("✅ Simulation complete! Switch to the **Results** or **Curves** tabs.")
                st.rerun()
            else:
                st.error("❌ Simulation failed. Check the log above.")

        except FileNotFoundError:
            st.error(f"Runner script not found: {runner_script}")
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        past_runs = list_past_runs()
        if past_runs:
            st.markdown("#### Load a past run")
            run_names = [f[0] for f in past_runs]
            selected_run = st.selectbox("Select a past run:", run_names, index=0)
            if st.button("Load selected run"):
                path = dict(past_runs)[selected_run]
                data = load_results_from_path(path)
                if data:
                    with open(LATEST_JSON, "w") as f:
                        json.dump({"path": path}, f)
                    st.success("Loaded. Switch to the Results tab.")
                    st.rerun()
        else:
            st.info(
                "No past runs found. Configure the experiment in the sidebar "
                "and click **▶ Run Experiment** to start."
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Experiment Results")

    results = load_latest_results()

    if results is None:
        st.info("No results yet. Run an experiment from the sidebar.")
    else:
        cfg = results["config"]
        summ = results["summary"]

        # ── Summary metrics ──
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Final Accuracy", f"{summ['final_accuracy'] * 100:.1f}%")
        m2.metric("Max Accuracy", f"{summ['max_accuracy'] * 100:.1f}%")
        m3.metric("Final Loss", f"{summ['final_loss']:.4f}")
        m4.metric("Rounds completed", summ["total_rounds"])
        m5.metric("Total time", f"{summ['total_time_sec']:.0f}s")

        converged = summ.get("converged", False)
        if converged:
            st.success("The model appears to have **converged** (accuracy stable in last 3 rounds).")
        else:
            st.info("The model has not converged yet — consider increasing the number of rounds.")

        st.divider()

        # ── Config recap ──
        st.markdown("#### Run configuration")
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
        cfg_col1.markdown(f"**Clients:** {cfg['num_clients']}")
        cfg_col2.markdown(f"**Rounds:** {cfg['num_rounds']}")
        cfg_col3.markdown(f"**fraction_fit:** {cfg['fraction_fit']}")
        cfg_col4.markdown(f"**fraction_evaluate:** {cfg['fraction_evaluate']}")

        cfg_col5, cfg_col6, cfg_col7, cfg_col8 = st.columns(4)
        cfg_col5.markdown(f"**Local epochs:** {cfg['local_epochs']}")
        cfg_col6.markdown(f"**Batch size:** {cfg['batch_size']}")
        cfg_col7.markdown(f"**Dataset:** {cfg['dataset']}")
        cfg_col8.markdown(f"**Seed:** {cfg['seed']}")

        st.divider()

        # ── Per-round table ──
        with st.expander("Per-round metrics table", expanded=True):
            rows = []
            for i, rnd in enumerate(results["rounds"]):
                rows.append({
                    "Round": rnd,
                    "Accuracy (%)": f"{results['global_accuracy'][i] * 100:.2f}",
                    "Loss": f"{results['global_loss'][i]:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CURVES
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Accuracy and Loss Curves")

    results = load_latest_results()

    if results is None:
        st.info("No results yet. Run an experiment first.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(plotly_accuracy(results), use_container_width=True)
            st.caption(
                "Each point corresponds to one communication round. "
                "The model is evaluated on the global test set after each round."
            )
        with col_r:
            st.plotly_chart(plotly_loss(results), use_container_width=True)
            st.caption(
                "Cross-entropy loss on the MNIST test set. "
                "A decreasing trend indicates the model is learning."
            )

        st.divider()

        # ── Accuracy improvement per round ──
        acc = results["global_accuracy"]
        if len(acc) > 1:
            st.markdown("#### Round-over-round accuracy improvement")
            deltas = [0.0] + [(acc[i] - acc[i - 1]) * 100 for i in range(1, len(acc))]
            rounds = results["rounds"]
            fig_delta = go.Figure()
            colors = ["#2e7d32" if d >= 0 else "#c62828" for d in deltas]
            fig_delta.add_trace(go.Bar(
                x=rounds, y=deltas,
                marker_color=colors,
                name="Accuracy change (%)",
                hovertemplate="Round %{x}: %{y:+.2f}%<extra></extra>",
            ))
            fig_delta.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1, opacity=0.5)
            fig_delta.update_layout(
                xaxis_title="Round",
                yaxis_title="Δ Accuracy (%)",
                title="Accuracy gain per round",
                template="plotly_white",
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_delta, use_container_width=True)
            st.caption(
                "Green: accuracy improved this round. Red: accuracy decreased. "
                "A flat trend in the later rounds suggests convergence."
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("Export ::Raw Results")

    results = load_latest_results()

    if results is None:
        st.info("No results yet. Run an experiment first.")
    else:
        cfg = results["config"]
        summ = results["summary"]

        st.markdown(
            f"**Current run:** {cfg['num_clients']} clients · {cfg['num_rounds']} rounds · "
            f"seed={cfg['seed']} · final accuracy {summ['final_accuracy'] * 100:.1f}%"
        )

        st.divider()

        # ── Download JSON ──
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="⬇ Download results as JSON",
            data=results_json,
            file_name=f"fedavg_results_seed{cfg['seed']}.json",
            mime="application/json",
        )

        st.divider()

        # ── Preview the JSON ──
        with st.expander("Preview JSON content"):
            st.json(results)

        st.divider()

        # ── CSV export of per-round metrics ──
        st.markdown("#### Per-round metrics (CSV)")
        rows = []
        for i, rnd in enumerate(results["rounds"]):
            rows.append({
                "round": rnd,
                "accuracy": results["global_accuracy"][i],
                "loss": results["global_loss"][i],
            })
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇ Download per-round CSV",
            data=csv_data,
            file_name=f"fedavg_rounds_seed{cfg['seed']}.csv",
            mime="text/csv",
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — FEDAVG NOTES
# Local Q&A helper — no external API required.
# ════════════════════════════════════════════════════════════════════════════

_FEDAVG_QA = {
    "what is fedavg": (
        "**FedAvg (Federated Averaging)** is the standard federated learning algorithm, "
        "introduced by McMahan et al. (2017).\n\n"
        "In each round:\n"
        "1. The server sends the global model to selected clients.\n"
        "2. Each client trains locally on its private data.\n"
        "3. Clients send their updated weights to the server.\n"
        "4. The server computes a weighted average of the client weights.\n"
        "5. This average becomes the new global model.\n\n"
        "The key insight is that **data never leaves the client**."
    ),
    "what is local training": (
        "**Local training** refers to the steps each client runs on its own data "
        "before sending updates to the server.\n\n"
        "In FedAvg, each client:\n"
        "- Receives the current global model weights.\n"
        "- Trains the model on its local dataset for `local_epochs` epochs.\n"
        "- Sends the resulting weights back to the server.\n\n"
        "More local epochs = fewer communication rounds needed, but higher risk "
        "of clients diverging from each other (client drift)."
    ),
    "what is fraction_fit": (
        "**fraction_fit** controls what fraction of clients participate in training "
        "each round.\n\n"
        "- `fraction_fit = 1.0` → all clients train every round.\n"
        "- `fraction_fit = 0.5` → half of the clients are randomly selected each round.\n\n"
        "Using a fraction < 1.0 can speed up each round but may slow convergence. "
        "It also simulates realistic settings where not all clients are available."
    ),
    "what is fraction_evaluate": (
        "**fraction_evaluate** controls what fraction of clients are used to evaluate "
        "the global model after each round.\n\n"
        "These clients evaluate the model on their local data and report metrics "
        "back to the server. The server then aggregates these metrics (e.g., weighted "
        "average accuracy)."
    ),
    "how many rounds": None,  # Answered dynamically from results
    "what is the final accuracy": None,  # Answered dynamically from results
    "what is the final loss": None,  # Answered dynamically from results
    "did the model converge": None,  # Answered dynamically from results
    "what happens when we increase rounds": (
        "**Increasing the number of rounds** generally:\n"
        "- Allows the global model to continue improving if it has not converged yet.\n"
        "- Increases total training time (each round requires client communication).\n"
        "- Eventually shows diminishing returns — the model converges and adding more "
        "rounds provides little additional accuracy gain.\n\n"
        "A good strategy is to plot the accuracy curve and stop when the improvement "
        "per round drops below a threshold (e.g., 0.5%)."
    ),
    "what happens when we increase clients": (
        "**Increasing the number of clients:**\n"
        "- More data distributed across the federation (can improve generalization).\n"
        "- Each round takes longer if fraction_fit = 1.0 (all clients must train).\n"
        "- With fraction_fit < 1.0, individual rounds stay fast but convergence may "
        "require more rounds.\n"
        "- In practice, more clients better simulate real-world FL deployments."
    ),
}


def _local_qa(message: str, results: Optional[dict]) -> str:
    msg = message.lower().strip()
    cfg = results.get("config", {}) if results else {}
    summ = results.get("summary", {}) if results else {}
    has_run = bool(results)

    # Dynamic questions that depend on loaded results
    if "final accuracy" in msg or "accuracy" in msg and "final" in msg:
        if not has_run:
            return "No run is loaded yet. Run an experiment first."
        return (
            f"The final accuracy of the loaded run is "
            f"**{summ.get('final_accuracy', 0) * 100:.1f}%** "
            f"(max across all rounds: **{summ.get('max_accuracy', 0) * 100:.1f}%**)."
        )

    if "final loss" in msg:
        if not has_run:
            return "No run is loaded yet. Run an experiment first."
        return f"The final loss of the loaded run is **{summ.get('final_loss', 0):.4f}**."

    if "converge" in msg or "converged" in msg:
        if not has_run:
            return "No run is loaded yet. Run an experiment first."
        if summ.get("converged"):
            return (
                "Based on the accuracy in the last 3 rounds, the model appears to have "
                "**converged** — the improvement per round dropped below 0.5%. "
                "You can try adding more rounds to confirm."
            )
        else:
            return (
                "The model does **not appear to have fully converged** yet. "
                f"Final accuracy is {summ.get('final_accuracy', 0)*100:.1f}% after "
                f"{summ.get('total_rounds', '?')} rounds. "
                "Try increasing the number of rounds."
            )

    if "how many rounds" in msg or "number of rounds" in msg:
        if not has_run:
            return "No run is loaded yet. Run an experiment first."
        return (
            f"The loaded run executed **{summ.get('total_rounds', '?')} rounds** "
            f"(configured: {cfg.get('num_rounds', '?')})."
        )

    # Static Q&A
    for key, answer in _FEDAVG_QA.items():
        if answer and key in msg:
            return answer

    # Fallback
    return (
        "I can answer questions about the loaded run or FedAvg concepts. Try:\n"
        "- *What is FedAvg?*\n"
        "- *What is fraction_fit?*\n"
        "- *Did the model converge?*\n"
        "- *What is the final accuracy?*\n"
        "- *What happens when we increase rounds?*\n"
        "- *What is local training?*"
    )


with tabs[5]:
    st.header("📝 FedAvg Notes")
    st.caption(
        "A local Q&A helper for understanding FedAvg concepts and inspecting results. "
        "No external API key required — all answers are generated locally."
    )

    _results = load_latest_results()

    if _results:
        _cfg = _results.get("config", {})
        _summ = _results.get("summary", {})
        st.success(
            f"Loaded run: **{_cfg.get('num_clients', '?')} clients**, "
            f"**{_cfg.get('num_rounds', '?')} rounds**, "
            f"final accuracy **{_summ.get('final_accuracy', 0) * 100:.1f}%**."
        )
    else:
        st.info("No run loaded yet. Run an experiment first, or use this tab to learn about FedAvg.")

    st.divider()

    # ── Quick questions ──
    st.markdown("**Quick questions**")
    quick_questions = [
        "Did the model converge?",
        "What is the final accuracy?",
        "What happens when we increase rounds?",
    ]

    q_cols = st.columns(3)
    for qi, q in enumerate(quick_questions):
        if q_cols[qi % 3].button(q, use_container_width=True, key=f"qa_btn_{qi}"):
            st.session_state.setdefault("notes_msgs", []).append({"role": "user", "content": q})
            answer = _local_qa(q, _results)
            st.session_state["notes_msgs"].append({"role": "assistant", "content": answer})
            st.rerun()

    st.divider()

    # ── Conversation ──
    st.markdown("### Ask a question")
    for msg in st.session_state.get("notes_msgs", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about FedAvg, the loaded run, or the parameters...")
    if user_q:
        st.session_state.setdefault("notes_msgs", []).append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        answer = _local_qa(user_q, _results)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["notes_msgs"].append({"role": "assistant", "content": answer})

    if st.session_state.get("notes_msgs"):
        if st.button("Clear chat", key="clear_notes"):
            st.session_state["notes_msgs"] = []
            st.rerun()

    st.divider()

    # ── FedAvg reference cards ──
    st.markdown("### Reference: FedAvg algorithm")
    with st.expander("FedAvg ::one round (step by step)"):
        st.markdown("""
        1. **Server selects clients** — a random subset of `fraction_fit × N` clients.
        2. **Server broadcasts** the current global model weights to selected clients.
        3. **Each client trains** the model locally for `local_epochs` epochs on its private data.
        4. **Clients send** their updated weights (and sample count) back to the server.
        5. **Server aggregates** using a weighted average:
           - weight for client *i* = (samples of client *i*) / (total samples across selected clients)
           - new global weights = Σ (client weight × client model weights)
        6. The averaged weights form the **new global model** → next round begins.
        """)

    with st.expander("Why does data stay local?"):
        st.markdown("""
        In FedAvg, only **model weights** (tensors of floating-point numbers) are sent to the server.
        The raw training data (images, text, etc.) never leaves the client machine.

        This provides a basic form of data privacy: the server cannot directly reconstruct
        the training samples from the model weights alone.

        Note: this does not guarantee full privacy — gradient-based inference attacks exist.
        But for many practical scenarios, keeping data local is sufficient for compliance.
        """)

    with st.expander("Tips for your first experiment"):
        st.markdown("""
        - **Start small:** 4 clients, 5 rounds, fraction_fit=1.0, local_epochs=1.
        - **Watch the accuracy curve:** if it is still rising at round 5, add more rounds.
        - **Try fraction_fit < 1.0** to see how partial participation affects convergence speed.
        - **Increase local_epochs** to reduce the number of communication rounds needed,
          at the cost of higher variance between clients.
        - **Compare seeds:** run the same config with seed=42 and seed=123 to see variance.
        """)
