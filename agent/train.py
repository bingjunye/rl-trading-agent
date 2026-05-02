"""
agent/train.py

Training script for the PPO trading agent.

Usage:
    python agent/train.py                          # use default config
    python agent/train.py --reward sharpe          # override reward type
    python agent/train.py --reward return          # ablation: raw return
    python agent/train.py --timesteps 1000000      # longer training
"""

import argparse
import yaml
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Make sure Python finds our modules regardless of working directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import download_data, train_test_split
from env.trading_env import TradingEnv


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(df, config, reward_type: str = None) -> TradingEnv:
    reward = reward_type or config["env"]["reward_type"]
    env = TradingEnv(
        df=df,
        window_size=config["env"]["window_size"],
        initial_balance=config["env"]["initial_balance"],
        transaction_cost=config["env"]["transaction_cost"],
        reward_type=reward,
    )
    return Monitor(env)


def train(config: dict, reward_type: str = None, timesteps: int = None):
    reward = reward_type or config["env"]["reward_type"]
    total_timesteps = timesteps or config["agent"]["total_timesteps"]
    run_name = f"ppo_{reward}"

    print(f"\n{'='*50}")
    print(f"  Training: {run_name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"{'='*50}\n")

    # Data
    df = download_data(
        config["data"]["ticker"],
        config["data"]["start_date"],
        config["data"]["end_date"],
    )
    train_df, eval_df = train_test_split(df, config["data"]["train_ratio"])

    # Environments
    train_env = make_env(train_df, config, reward_type=reward)
    eval_env  = make_env(eval_df,  config, reward_type=reward)

    # Callbacks
    results_dir = Path("results") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(results_dir),
        log_path=str(results_dir),
        eval_freq=10_000,
        n_eval_episodes=1,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(results_dir / "checkpoints"),
        name_prefix="ppo",
    )

    # Model — hyperparameters from config
    ag = config["agent"]
    model = PPO(
        policy=ag["policy"],
        env=train_env,
        learning_rate=ag["learning_rate"],
        n_steps=ag["n_steps"],
        batch_size=ag["batch_size"],
        n_epochs=ag["n_epochs"],
        gamma=ag["gamma"],
        ent_coef=ag["ent_coef"],
        clip_range=ag["clip_range"],
        tensorboard_log=f"results/tensorboard/{run_name}",
        verbose=ag["verbose"],
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=run_name,
    )

    model.save(str(results_dir / "final_model"))
    print(f"\nModel saved to results/{run_name}/")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--reward", choices=["return", "sharpe", "sortino"], default=None,
                        help="Override reward type from config (useful for ablation study)")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, reward_type=args.reward, timesteps=args.timesteps)
