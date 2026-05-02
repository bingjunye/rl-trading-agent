"""
utils/features.py

Feature engineering for the trading environment.
This is where your DS background adds value — thoughtful feature selection
matters more than the RL algorithm itself.

Design principle: all features are normalized to similar scales,
so the neural network doesn't have to learn unit conversions.
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and return a clean feature DataFrame.

    Features are chosen to give the agent information about:
    - Momentum (are we trending?)
    - Mean reversion (are we overextended?)
    - Volatility (how risky is the current environment?)
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # --- Momentum: price returns at multiple horizons ---
    for window in [1, 5, 10, 20]:
        feat[f"return_{window}d"] = close.pct_change(window)

    # --- RSI: overbought/oversold signal (normalized to [-1, 1]) ---
    feat["rsi_14"] = _rsi(close, period=14) / 50 - 1

    # --- MACD: trend direction ---
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    feat["macd_hist"] = (macd_line - macd_signal) / close  # normalized by price

    # --- Bollinger Band position: where in the band are we? ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_position"] = (close - sma20) / (2 * std20)  # ~[-1, 1] typically

    # --- Volatility: rolling realized vol (annualized) ---
    feat["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # --- Volume: normalized volume (z-score) ---
    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    feat["volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-8)

    # Drop rows with NaN from rolling windows
    feat = feat.dropna()
    return feat


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def get_feature_names() -> list[str]:
    return [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "rsi_14", "macd_hist", "bb_position", "vol_20d", "volume_zscore",
    ]


if __name__ == "__main__":
    from data_loader import download_data
    from pathlib import Path
    import yaml

    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = download_data(config["data"]["ticker"], config["data"]["start_date"], config["data"]["end_date"])
    features = compute_features(df)
    print(features.tail())
    print(f"\nFeature shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
