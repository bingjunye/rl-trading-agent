# Risk-Aware Stock Trading Agent with PPO

## 我們想解決什麼問題

傳統的股票交易策略有兩個根本缺陷。

規則型策略（例如移動平均線）的參數是固定的，市場從趨勢變成震盪的時候，策略就會持續失效，直到人工介入調整。監督式學習可以預測漲跌方向，但無法回答「現在應該持多少倉位」這個問題，因為這個決策依賴於你目前的持倉狀態、未實現損益、以及當前的市場風險，是一個有時間連動性的序列決策問題。

我們想解決的核心問題是：**在不確定的市場環境下，能不能學出一個動態的交易策略，在控制下行風險的同時，長期打敗 Buy & Hold？**

## 為什麼用 RL

**決策有序列依賴性。** 今天買進之後，明天的選擇就受限了。這種「當前決策影響未來狀態」的結構，監督式學習處理不了。

**Reward 是延遲的。** 今天買進，不知道幾天後才知道這個決策是對的。RL 天生為延遲 reward 設計。

**需要在不同市場狀態下有不同行為。** 牛市應該持有，熊市應該觀望，震盪市應該減少交易頻率。這種條件性行為需要 RL 的 exploration 機制來學習。

## 方法

用 PPO 訓練一個 agent，每天觀察市場狀態，決定買進、持有或賣出。

**State：** 過去 20 天的技術指標，包含價格動能、RSI、MACD、布林帶位置、波動率、成交量，加上 agent 當前的持倉狀態和未實現損益。

**Action：** 買進 / 持有 / 賣出

**Reward（核心實驗變因）：**
- `return`：raw daily return，只看報酬
- `sharpe`：報酬除以波動，懲罰不穩定
- `sortino`：報酬除以下行波動，只懲罰虧損

## 數據

SPY（標普500 ETF），2015/01/01 至 2025/01/01，共 2264 個交易日。按時間順序切分：前 80%（2015-2022）訓練，後 20%（2022-2024）測試。測試集涵蓋 2022 聯準會升息熊市、2023 反彈、2024 牛市。

## 實驗結果

| 策略 | Sharpe | Max Drawdown | CAGR | Total Return |
|---|---|---|---|---|
| Buy & Hold | 0.371 | -22.7% | 7.6% | 14.0% |
| SMA Crossover | -0.228 | -18.3% | -1.0% | -1.7% |
| PPO (return reward) | 0.624 | -18.2% | 10.0% | 18.7% |
| **PPO (sharpe reward)** | **1.419** | **-10.3%** | **21.4%** | **41.6%** |
| PPO (sortino reward) | 0.071 | -15.4% | 2.0% | 3.6% |

![Equity Curves](results/figures/equity_curves.png)

## Key Findings

**Finding 1：Reward 設計對結果影響極大。** 同樣的 PPO 演算法，sharpe reward 的 CAGR 是 21.4%，sortino reward 只有 2.0%，差了十倍。RL 在交易上的成敗，reward 的設計比演算法本身更關鍵。

**Finding 2：Sharpe reward 在熊市中保護了下行。** 2022 年大跌期間，sharpe reward agent 的 max drawdown 只有 -10.3%，而 Buy & Hold 是 -22.7%。Sharpe reward 讓 agent 學會「波動大的時候不進場」，在熊市自然轉向保守。

**Finding 3：Sortino reward 數值不穩定。** 訓練 log 顯示 value_loss 在後期爆炸到 2.3e+08，explained_variance 接近 0。Sortino 的非對稱結構產生極度偏斜的 reward 分布，PPO 的 value network 難以收斂。

## Limitations & Future Work

- 只跑了一個 random seed，需要多次實驗排除運氣成分
- Action space 是離散的，無法做部分倉位調整
- Sortino reward 失敗可能可以透過 reward normalization 修復
- 未來可嘗試 reward = 報酬率 - λ × 波動率，更直接對應投資人目標

## Project Structure
rl-trading-agent/
├── configs/default.yaml      # 所有超參數集中管理
├── env/trading_env.py        # 自訂 Gymnasium 環境
├── agent/train.py            # PPO 訓練（支援 --reward 切換）
├── utils/
│   ├── data_loader.py        # 數據載入
│   ├── features.py           # 技術指標 feature engineering
│   └── metrics.py            # Sharpe, Sortino, Max Drawdown, CAGR
├── evaluate.py               # 回測與 baseline 比較
└── results/figures/          # 產出圖表

## Quickstart

```bash
pip install -r requirements.txt

# 訓練（三種 reward 各一次）
python agent/train.py --reward sharpe
python agent/train.py --reward return
python agent/train.py --reward sortino

# 比較結果
python evaluate.py --compare
```

## Tech Stack

`gymnasium` · `stable-baselines3` · `pandas` · `pandas-ta` · `matplotlib` · `tensorboard`

---

## 常見疑問

**Q：為什麼用 RL，不用其他方法？**

交易是一個序列決策問題，今天買了之後明天的選擇就受限了，而且 reward 是延遲的，不知道今天的決策幾天後才知道對不對。這兩個特性讓監督式學習處理不了，RL 天生就是為這種問題設計的。

**Q：Reward 怎麼設計的？**

測試了三種。純報酬率最直覺但容易過擬合牛市。Sharpe ratio 把報酬除以波動，強迫 agent 同時控制風險，結果在真實市場表現最穩。Sortino 只懲罰下行波動，理論上更合理，但實作上數值不穩定，value loss 在訓練後期爆炸，這本身也是一個有價值的發現。

**Q：結果可信嗎？有什麼 Limitation？**

只跑了一個 random seed，需要多次實驗才能排除運氣。Action space 是離散的，無法做部分倉位調整。只用了一支 ETF，泛化能力還沒驗證。這些都是誠實的 limitation，詳見上方 Limitations 章節。

**Q：為什麼選 PPO 而不是 DQN？**

我們的 action space 雖然是離散的，但 PPO 在訓練穩定性上優於 DQN，而且未來如果要擴展成連續的倉位比例（0% 到 100% 的 allocation），PPO 可以直接支援，DQN 需要重新設計。
