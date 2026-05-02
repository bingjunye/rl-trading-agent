"""
env/trading_env.py

Custom Gymnasium environment for stock trading.

Design decisions documented here (important for GitHub README and interviews):

1. ACTION SPACE: Discrete {0: hold, 1: buy, 2: sell}
   - Kept discrete for interpretability and training stability
   - A continuous "position sizing" extension is straightforward (see comments)

2. STATE SPACE: [technical_features..., current_position, unrealized_pnl]
   - Including current_position is critical — the agent needs to know what it holds
   - Including unrealized_pnl incentivizes appropriate exit timing

3. REWARD: Configurable (return / sharpe / sortino)
   - This is the key experimental variable in our ablation study
   - Sharpe reward uses a rolling window to estimate vol at training time

4. TRANSACTION COST: Flat 0.1% per trade
   - Prevents the agent from churning trades for tiny gains
   - Realistic for retail brokers
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from utils.features import compute_features, get_feature_names


class TradingEnv(gym.Env):
    """
    Single-asset trading environment.

    Observation: [technical features (window_size × n_features), position, unrealized_pnl]
    Action:      Discrete(3) — 0: hold, 1: buy, 2: sell
    Reward:      Configurable — daily return, Sharpe increment, or Sortino increment
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 10_000,
        transaction_cost: float = 0.001,
        reward_type: str = "sharpe",  # "return" | "sharpe" | "sortino"
    ):
        super().__init__()

        self.features = compute_features(df)
        self.prices = df["Close"].squeeze().loc[self.features.index]
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type

        n_features = len(get_feature_names())

        # Observation: flattened window of features + position + unrealized pnl
        obs_size = window_size * n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: hold / buy / sell
        self.action_space = spaces.Discrete(3)

        # Internal state (reset on each episode)
        self._step_idx = None
        self._balance = None
        self._shares = None
        self._entry_price = None
        self._returns_history = []  # used for rolling Sharpe reward

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = self.window_size
        self._balance = self.initial_balance
        self._shares = 0.0
        self._entry_price = 0.0
        self._returns_history = []
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        price = float(self.prices.iloc[self._step_idx])
        prev_portfolio = self._portfolio_value(price)

        # Execute action
        if action == 1 and self._shares == 0:   # Buy
            cost = price * (1 + self.transaction_cost)
            self._shares = self._balance / cost
            self._balance = 0.0
            self._entry_price = price

        elif action == 2 and self._shares > 0:  # Sell
            proceeds = self._shares * price * (1 - self.transaction_cost)
            self._balance = proceeds
            self._shares = 0.0
            self._entry_price = 0.0

        # Move to next step
        self._step_idx += 1
        terminated = self._step_idx >= len(self.prices) - 1

        new_price = float(self.prices.iloc[self._step_idx])
        new_portfolio = self._portfolio_value(new_price)

        # Compute reward
        daily_return = (new_portfolio - prev_portfolio) / (prev_portfolio + 1e-8)
        self._returns_history.append(daily_return)
        reward = self._compute_reward(daily_return)

        obs = self._get_obs()
        info = {
            "portfolio_value": new_portfolio,
            "position": "long" if self._shares > 0 else "flat",
            "daily_return": daily_return,
        }
        return obs, reward, terminated, False, info

    def render(self, mode="human"):
        price = float(self.prices.iloc[self._step_idx])
        portfolio = self._portfolio_value(price)
        print(f"Step {self._step_idx} | Price: {price:.2f} | Portfolio: {portfolio:.2f} | "
              f"Shares: {self._shares:.4f}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        window = self.features.iloc[self._step_idx - self.window_size : self._step_idx]
        feature_vec = window.values.flatten().astype(np.float32)

        price = float(self.prices.iloc[self._step_idx])
        position = float(self._shares > 0)
        unrealized_pnl = (
            (price - self._entry_price) / (self._entry_price + 1e-8)
            if self._shares > 0 else 0.0
        )
        return np.concatenate([feature_vec, [position, unrealized_pnl]], dtype=np.float32)

    def _portfolio_value(self, price: float) -> float:
        return self._balance + self._shares * price

    def _compute_reward(self, daily_return: float) -> float:
        """
        Reward function — this is the key experimental variable.

        "return"  : raw daily return (baseline)
        "sharpe"  : rolling Sharpe ratio increment
        "sortino" : rolling Sortino ratio increment (penalizes only downside vol)
        """
        if self.reward_type == "return":
            return daily_return

        returns = np.array(self._returns_history)
        if len(returns) < 5:
            return daily_return  # not enough history yet, fall back to return

        if self.reward_type == "sharpe":
            vol = returns.std() + 1e-8
            return returns.mean() / vol  # rolling Sharpe proxy

        if self.reward_type == "sortino":
            downside = returns[returns < 0]
            down_vol = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
            return returns.mean() / down_vol

        raise ValueError(f"Unknown reward_type: {self.reward_type}")
