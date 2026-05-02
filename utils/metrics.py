"""
utils/metrics.py

Evaluation metrics for trading strategies.
These metrics are what you report in the README and compare across experiments.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Annualized Sharpe ratio.
    risk_free_rate: annual rate (e.g. 0.02 = 2%)
    """
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Like Sharpe, but only penalizes downside volatility.
    Better measure for asymmetric return distributions.
    """
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / downside.std()


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough decline.
    Returns a negative number (e.g. -0.25 = 25% drawdown).
    """
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def cagr(equity_curve: np.ndarray, n_years: float) -> float:
    """Compound Annual Growth Rate."""
    if n_years <= 0 or equity_curve[0] <= 0:
        return 0.0
    return (equity_curve[-1] / equity_curve[0]) ** (1 / n_years) - 1


def summarize(equity_curve: np.ndarray, n_years: float, label: str = "") -> dict:
    """Print and return a full performance summary."""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    metrics = {
        "label": label,
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "cagr": cagr(equity_curve, n_years),
        "total_return": (equity_curve[-1] / equity_curve[0]) - 1,
    }
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.3f}")
    print(f"  Sortino Ratio:  {metrics['sortino']:.3f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.1%}")
    print(f"  CAGR:           {metrics['cagr']:.1%}")
    print(f"  Total Return:   {metrics['total_return']:.1%}")
    return metrics
