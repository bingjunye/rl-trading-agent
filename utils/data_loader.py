"""
utils/data_loader.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Load from local cache. Supports Google Sheets exported CSV."""
    
    # 優先找 Google Sheets 下載的檔案
    google_path = DATA_DIR / f"{ticker}_google.csv"
    if google_path.exists():
        print(f"Loading {ticker} from Google Sheets CSV: {google_path}")
        df = pd.read_csv(google_path)
        
        # 清理日期欄位（移除「下午 4:00:00」之類的時間部分）
        df["Date"] = df["Date"].str.split(" ").str[0]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df.sort_index()
        
        # 轉成 MultiIndex columns，跟 yfinance 格式一致
        df.columns = pd.MultiIndex.from_tuples([(col, ticker) for col in df.columns])
        df = df[(df.index >= start) & (df.index <= end)]
        df = df.dropna()
        return df

    print(f"No local data found for {ticker}. Please download manually.")
    raise FileNotFoundError(f"data/{ticker}_google.csv not found")


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)")
    return train, test


if __name__ == "__main__":
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    df = download_data(config["data"]["ticker"], config["data"]["start_date"], config["data"]["end_date"])
    print(f"Shape: {df.shape}")
    print(df.head(2))
