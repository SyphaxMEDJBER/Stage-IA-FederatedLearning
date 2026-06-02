# FLows — FedAvg Intern Dashboard

**Simplified Flower/FedAvg dashboard for L3 internship onboarding**

This is a standalone, self-contained dashboard for learning standard Federated Averaging (FedAvg).
It is **completely separate** from the main ClusFed/ACFRO thesis dashboard (`../app.py`).

---

## Quick start

```bash
cd poc_thesis/intern_fedavg_dashboard

# Install dependencies (if not already installed from the parent project)
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dashboard pages

| Tab | Content |
|---|---|
| 📖 Overview | What FedAvg is, how it works, key parameters |
| 🚀 Run Experiment | Configure and launch a FedAvg run on MNIST |
| 📊 Results | Final/max accuracy, loss, per-round table |
| 📈 Curves | Accuracy and loss plots per round |
| 💾 Export | Download results as JSON or CSV |
| 📝 FedAvg Notes | Local Q&A helper — no API key required |

---

## Configuration parameters

| Parameter | Description |
|---|---|
| `num_clients` | Total number of federated clients |
| `num_rounds` | Number of communication rounds |
| `fraction_fit` | Fraction of clients selected for training each round |
| `fraction_evaluate` | Fraction of clients used for evaluation |
| `local_epochs` | Local training epochs per client per round |
| `batch_size` | Mini-batch size |
| `seed` | Random seed for reproducibility |

---

## Notes

- No external API key required.
- No ClusFed, ACFRO, clustering, or adversarial client logic.
- Results are saved in `outputs/runs/` as JSON files.
- The original ClusFed dashboard (`../app.py`) is completely untouched.
