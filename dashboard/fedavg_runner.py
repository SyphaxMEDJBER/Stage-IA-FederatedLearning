"""
fedavg_runner.py
----------------
FedAvg simulation runner for intern onboarding.
Uses standard Flower FedAvg strategy on MNIST — no clustering, no attacks.

Usage:
    python fedavg_runner.py [--clients N] [--rounds R] [--fraction_fit F]
                            [--fraction_evaluate E] [--local_epochs L]
                            [--batch_size B] [--seed S] [--output PATH]

Outputs:
    - JSON results saved to --output path
    - Progress lines to stdout (captured by the Streamlit app)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional

# Suppress TF and Ray noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("RAY_memory_usage_threshold", "0.98")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import flwr as fl
from flwr.common import ndarrays_to_parameters


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_model() -> tf.keras.Model:
    """Simple MLP for MNIST: Flatten → Dense(128) → Dropout → Dense(10)."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist_partitions(num_clients: int, seed: int):
    """Load MNIST and split it evenly across clients (IID partition)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    partition_size = len(x_train) // num_clients
    partitions = []
    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size
        partitions.append((x_train[start:end], y_train[start:end]))

    return partitions, (x_test, y_test)


# ---------------------------------------------------------------------------
# Flower client
# ---------------------------------------------------------------------------

class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_val, y_val, local_epochs: int, batch_size: int):
        self.model = get_model()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        return float(loss), len(self.x_val), {"accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Instrumented FedAvg strategy (records per-round metrics)
# ---------------------------------------------------------------------------

class InstrumentedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds: int, progress_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rounds: List[int] = []
        self.accuracies: List[float] = []
        self.losses: List[float] = []
        self.num_rounds = num_rounds
        self._cb = progress_callback or print

    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        if isinstance(agg, tuple) and len(agg) == 2:
            loss, metrics = agg
            accuracy = float(metrics.get("accuracy", 0.0))
            self.rounds.append(server_round)
            self.accuracies.append(accuracy)
            self.losses.append(float(loss))
            self._cb(
                f"[Round {server_round}/{self.num_rounds}] "
                f"Accuracy: {accuracy * 100:.2f}%  Loss: {loss:.4f}"
            )
        return agg


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def _check_convergence(acc: list, window: int = 3, threshold: float = 0.005) -> bool:
    if len(acc) < window + 1:
        return False
    recent = acc[-(window + 1):]
    return (max(recent) - min(recent)) < threshold


def run_simulation(
    num_clients: int,
    num_rounds: int,
    fraction_fit: float,
    fraction_evaluate: float,
    local_epochs: int,
    batch_size: int,
    seed: int,
    progress_callback=None,
) -> dict:
    cb = progress_callback or print

    tf.random.set_seed(seed)
    np.random.seed(seed)

    cb(f"[Runner] Loading MNIST — creating {num_clients} client partitions...")
    partitions, (x_val, y_val) = load_mnist_partitions(num_clients, seed)

    def client_fn(cid: str) -> fl.client.Client:
        x_tr, y_tr = partitions[int(cid)]
        return FedAvgClient(x_tr, y_tr, x_val, y_val, local_epochs, batch_size).to_client()

    initial_params = ndarrays_to_parameters(get_model().get_weights())

    min_fit = max(1, int(num_clients * fraction_fit))
    min_eval = max(1, int(num_clients * fraction_evaluate))

    strategy = InstrumentedFedAvg(
        num_rounds=num_rounds,
        progress_callback=cb,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit,
        min_evaluate_clients=min_eval,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
    )

    cb(f"[Runner] Starting Flower FedAvg simulation ({num_rounds} rounds, {num_clients} clients)...")
    start = time.time()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={
            "include_dashboard": False,
            "num_cpus": min(num_clients, os.cpu_count() or 2),
        },
    )

    total_time = time.time() - start
    acc = strategy.accuracies
    loss = strategy.losses

    return {
        "config": {
            "dataset": "MNIST",
            "strategy": "FedAvg",
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "fraction_fit": fraction_fit,
            "fraction_evaluate": fraction_evaluate,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
        "rounds": strategy.rounds,
        "global_accuracy": acc,
        "global_loss": loss,
        "summary": {
            "final_accuracy": acc[-1] if acc else 0.0,
            "max_accuracy": max(acc) if acc else 0.0,
            "final_loss": loss[-1] if loss else 0.0,
            "total_rounds": len(strategy.rounds),
            "total_time_sec": round(total_time, 1),
            "converged": _check_convergence(acc),
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FedAvg simulation on MNIST.")
    p.add_argument("--clients",           type=int,   default=4)
    p.add_argument("--rounds",            type=int,   default=5)
    p.add_argument("--fraction_fit",      type=float, default=1.0)
    p.add_argument("--fraction_evaluate", type=float, default=1.0)
    p.add_argument("--local_epochs",      type=int,   default=1)
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--output",            type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs", "runs"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}_fedavg_results.json")

    print(f"[Runner] Output: {output_path}", flush=True)

    results = run_simulation(
        num_clients=args.clients,
        num_rounds=args.rounds,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        progress_callback=lambda msg: print(msg, flush=True),
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"[Runner] Saved results to: {output_path}", flush=True)

    # Update latest.json pointer so the dashboard can auto-load
    latest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", "latest.json"
    )
    with open(latest_path, "w") as f:
        json.dump({"path": os.path.abspath(output_path)}, f)

    print(f"[Runner] Done.", flush=True)


if __name__ == "__main__":
    main()
