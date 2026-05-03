"""
evaluate.py

Backtests a trained model and compares it against baselines.
Generates the equity curve plots and performance table for the README.

Usage:
    python evaluate.py --model results/ppo_sharpe/best_model
    python evaluate.py --compare  # run all reward types and compare
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from stable_baselines3 import PPO

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import download_data, train_test_split
from utils.metrics import summarize
from env.trading_env import TradingEnv


def load_config(path="configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Baseline strategies
# ------------------------------------------------------------------

def baseline_buy_and_hold(prices: pd.Series, initial_balance: float) -> np.ndarray:
    shares = initial_balance / float(prices.iloc[0])
    return (prices * shares).values


def baseline_sma_crossover(
    prices: pd.Series,
    initial_balance: float,
    fast: int = 20,
    slow: int = 50,
    cost: float = 0.001,
) -> np.ndarray:
    """Simple moving average crossover. Buy when fast > slow, sell otherwise."""
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = (sma_fast > sma_slow).fillna(False)

    balance = initial_balance
    shares = 0.0
    equity = []

    for i, (price, in_market) in enumerate(zip(prices, signal)):
        price = float(price)
        if in_market and shares == 0 and balance > 0:
            shares = balance / (price * (1 + cost))
            balance = 0.0
        elif not in_market and shares > 0:
            balance = shares * price * (1 - cost)
            shares = 0.0
        equity.append(balance + shares * price)

    return np.array(equity)


# ------------------------------------------------------------------
# RL agent backtest
# ------------------------------------------------------------------

def backtest_agent(model_path: str, df: pd.DataFrame, config: dict, reward_type: str) -> np.ndarray:
    env = TradingEnv(
        df=df,
        window_size=config["env"]["window_size"],
        initial_balance=config["env"]["initial_balance"],
        transaction_cost=config["env"]["transaction_cost"],
        reward_type=reward_type,
    )
    model = PPO.load(model_path)
    obs, _ = env.reset()
    equity = [config["env"]["initial_balance"]]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        equity.append(info["portfolio_value"])
        done = terminated or truncated

    return np.array(equity)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_equity_curves(curves: dict, save_path: str = "results/figures/equity_curves.png"):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "Buy & Hold": "#6b7280",
        "SMA Crossover": "#f59e0b",
        "PPO (return reward)": "#3b82f6",
        "PPO (sharpe reward)": "#10b981",
        "PPO (sortino reward)": "#f43f5e",
    }
    styles = {
        "Buy & Hold": "--",
        "SMA Crossover": "-.",
    }

    for label, curve in curves.items():
        normalized = curve / curve[0]  # normalize to 1.0 at start
        ax.plot(
            normalized,
            label=label,
            color=colors.get(label, "#374151"),
            linestyle=styles.get(label, "-"),
            linewidth=2 if "PPO" in label else 1.5,
            alpha=0.9,
        )

    ax.set_title("Out-of-Sample Equity Curves (normalized to 1.0)", fontsize=14, pad=12)
    ax.set_xlabel("Trading Days")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved equity curve → {save_path}")
    plt.show()


def plot_metrics_table(all_metrics: list, save_path: str = "results/figures/metrics_table.png"):
    """Render a clean comparison table as an image for the README."""
    fig, ax = plt.subplots(figsize=(10, len(all_metrics) * 0.6 + 1))
    ax.axis("off")

    headers = ["Strategy", "Sharpe", "Sortino", "Max Drawdown", "CAGR", "Total Return"]
    rows = [
        [
            m["label"],
            f"{m['sharpe']:.3f}",
            f"{m['sortino']:.3f}",
            f"{m['max_drawdown']:.1%}",
            f"{m['cagr']:.1%}",
            f"{m['total_return']:.1%}",
        ]
        for m in all_metrics
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved metrics table → {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Path to a single trained model")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all reward types (needs all 3 models trained)")
    args = parser.parse_args()

    config = load_config(args.config)
    df = download_data(config["data"]["ticker"], config["data"]["start_date"], config["data"]["end_date"])
    _, test_df = train_test_split(df, config["data"]["train_ratio"])
    test_prices = test_df["Close"].squeeze()
    n_years = len(test_df) / 252

    initial = config["env"]["initial_balance"]
    curves = {}
    all_metrics = []

    # Baselines
    bh = baseline_buy_and_hold(test_prices, initial)
    curves["Buy & Hold"] = bh
    all_metrics.append(summarize(bh, n_years, "Buy & Hold"))

    sma = baseline_sma_crossover(test_prices, initial)
    curves["SMA Crossover"] = sma
    all_metrics.append(summarize(sma, n_years, "SMA Crossover"))

    # RL agent(s)
    reward_types = ["return", "sharpe", "sortino"] if args.compare else []
    if args.model:
        reward_types = []
        label = f"PPO ({Path(args.model).parent.name})"
        curve = backtest_agent(args.model, test_df, config, config["env"]["reward_type"])
        curves[label] = curve
        all_metrics.append(summarize(curve, n_years, label))

    for reward in reward_types:
        model_path = f"results/ppo_{reward}/best_model"
        if not Path(model_path + ".zip").exists():
            print(f"  Skipping {reward} — model not found at {model_path}")
            continue
        label = f"PPO ({reward} reward)"
        curve = backtest_agent(model_path, test_df, config, reward)
        curves[label] = curve
        all_metrics.append(summarize(curve, n_years, label))

    plot_equity_curves(curves)
    plot_metrics_table(all_metrics)
