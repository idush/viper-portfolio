# 🐍 Viper ML Trading Bot v3.0 - Enterprise Edition

**Professional Machine Learning Trading Bot for PocketOption Binary Options**

> Built with XGBoost + LSTM + CatBoost ensemble, multi-timeframe analysis, online learning, and adaptive risk management.

---

## 📋 Table of Contents

1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Guide](#configuration-guide)
6. [How to Use](#how-to-use)
7. [Common Errors & Fixes](#common-errors--fixes)
8. [Advanced Usage](#advanced-usage)
9. [Architecture](#architecture)
10. [FAQ](#faq)

---

## ✨ Features

### 🤖 AI & ML
- **Ensemble Model**: XGBoost + CatBoost + Random Forest + LSTM neural network
- **100+ Features**: Multi-timeframe indicators (1m, 5m, 15m)
- **Online Learning**: Model auto-retrains from live trades
- **Dynamic Confidence**: Adaptive prediction thresholds

### 📊 Trading Logic
- **Multi-Timeframe Analysis**: Scans 1m, 5m, 15m candles simultaneously
- **Feature Engineering**: Trend, momentum, volatility, price action features
- **Demo/Real Switching**: Test in demo, deploy to real with one click
- **Risk Management**: Kelly Criterion position sizing + volatility adjustment

### 🛡️ Risk Control
- **Dynamic Position Sizing**: Based on confidence and ATR
- **Max Loss Protection**: Stops trading if daily loss limit hit
- **Loss Streak Detection**: Pauses after 5 consecutive losses
- **Equity Curve Monitoring**: Tracks balance history

### 📈 Backtesting
- **Walk-Forward Validation**: Prevents overfitting
- **Multi-Day Backtest**: Test across weeks of data
- **Live Trade Logging**: Every trade recorded with reason
- **Performance Analytics**: Win rate, P/L, equity curves

---

## 💻 System Requirements

### Minimum
- **Python**: 3.8+
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **RAM**: 4GB
- **CPU**: Dual-core or better
- **Internet**: Stable connection to PocketOption

### Recommended
- **Python**: 3.10+
- **RAM**: 8GB+
- **GPU**: NVIDIA CUDA-capable GPU (optional, for faster LSTM training)

---

## 🚀 Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Navigate to your project directory
cd /path/to/binary-bot

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import xgboost; import tensorflow; import pocketoptionapi; print('✓ All dependencies installed')"
```

**Expected Output:**
```
✓ All dependencies installed
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Get Your Credentials

1. Open **PocketOption** website
2. Go to **Account Settings** → **API/SSID**
3. Copy your **SSID** (looks like: `c2gknk1hbndqt78dpv6eevk49j`)

### Step 2: Run the Bot

```bash
python viper_ml_bot.py
```

### Step 3: Configure & Backtest

```
============================================================
🐍 VIPER ML TRADING BOT v3.0 - PRODUCTION READY
============================================================
1. Configure Settings
2. Backtest
3. Show Configuration
4. Exit
============================================================
Select option: 1
```

**Follow the prompts:**
```
Symbol (current: None) [GBPJPY_otc]: GBPJPY_otc
SSID (current: None) [paste]: c2gknk1hbndqt78dpv6eevk49j
Mode (d=demo, r=real) [d]: d
Start day [10]: 16
End day [22]: 17
Target profit ($) [100]: 100
Max loss ($) [50]: 20
Base trade amount ($) [2]: 2
Confidence threshold (0-1) [0.55]: 0.55
✓ Configuration saved
```

### Step 4: Backtest

```
Select option: 2
```

**Watch the backtest run:**
```
📥 Downloading 3 TFs over 2 days...
✓ Downloaded: [(60, 1440), (300, 288), (900, 96)]

🧠 Training ensemble on 1200 samples...
✓ All models trained successfully

📊 BACKTEST RESULTS

#  1 15:22 CALL Conf=62% Size=$2.00 ✅ Bal=$1.84
#  2 15:28 PUT  Conf=58% Size=$2.00 ❌ Bal=$-0.16
...

✓ Final: 45 trades, 28W/17L, WR: 62.2%
  P/L: $45.78, Balance: $1045.78
```

---

## ⚙️ Configuration Guide

### Main Settings

| Setting | Default | Range | Meaning |
|---------|---------|-------|---------|
| **Symbol** | GBPJPY_otc | Any OTC pair | Currency pair to trade |
| **Mode** | demo | demo/real | Demo = no real money |
| **Start Day** | 10 | 1-31 | Backtest start (August) |
| **End Day** | 22 | 1-31 | Backtest end (August) |
| **Target Profit** | $100 | Any | Daily profit goal |
| **Max Loss** | $50 | Any | Stop if loss exceeds this |
| **Base Amount** | $2 | 0.1-100 | Trade size in dollars |
| **Confidence** | 0.55 | 0.5-1.0 | Min model confidence to trade |

### Available Symbols

```
GBPJPY_otc  - Most active, high volatility
EURUSD_otc  - Good for learning
USDCHF_otc  - Low correlation
AUDUSD_otc  - Asian session active
NZDUSD_otc  - Lower spreads
```

### Mode: Demo vs Real

**DEMO MODE** (Recommended for learning):
- No real money at risk
- Uses simulated PocketOption demo server
- Perfect for testing strategies
- Use: `Mode = demo`

**REAL MODE** (After proving profitability):
- Trades with real money
- Real P&L affects your account
- Start small: $1-5 per trade
- Use: `Mode = real` (⚠️ WARNING: Risk of losing money)

---

## 📖 How to Use

### Workflow

```
1. Install dependencies (requirements.txt)
   ↓
2. Get SSID from PocketOption
   ↓
3. Run bot: python viper_ml_bot.py
   ↓
4. Select "Configure Settings" (Option 1)
   ↓
5. Fill in your settings
   ↓
6. Select "Backtest" (Option 2)
   ↓
7. Wait for results
   ↓
8. Check logs in viper_bot.log
```

### Understanding Backtest Output

```
#  1 15:22 CALL Conf=62% Size=$2.00 ✅ Bal=$1.84
│   │    │    │      │    │         │   │
│   │    │    │      │    │         │   └─ New balance
│   │    │    │      │    │         └───── Trade result (✅ Win, ❌ Loss)
│   │    │    │      │    └───────────── Position size in dollars
│   │    │    │      └───────────────── Model confidence %
│   │    │    └────────────────────── Trade direction (CALL or PUT)
│   │    └───────────────────────── Trade time (HH:MM)
│   └──────────────────────────── Trade number
└─────────────────────────────── Trade ID
```

### Interpreting Results

```
✓ Final: 45 trades, 28W/17L, WR: 62.2%
  P/L: $45.78, Balance: $1045.78
```

- **45 trades**: Model triggered 45 times
- **28W/17L**: 28 wins, 17 losses
- **WR: 62.2%**: Win rate (target: 55%+)
- **P/L: $45.78**: Total profit/loss
- **Balance**: Starting $1000 + $45.78 = $1045.78

---

## 🔧 Common Errors & Fixes

### ❌ Error 1: "ModuleNotFoundError: No module named 'pocketoptionapi'"

**Cause**: Dependencies not installed

**Fix**:
```bash
pip install -r requirements.txt
```

---

### ❌ Error 2: "SSID not set"

**Cause**: You didn't enter SSID or left it blank

**Fix**:
```
1. Go to PocketOption → Account Settings → API
2. Copy your SSID (long alphanumeric string)
3. When prompted, paste it exactly
4. Don't include spaces or quotes
```

---

### ❌ Error 3: "Connection failed: 401 Unauthorized"

**Cause**: Invalid or expired SSID

**Fix**:
```bash
# Get a fresh SSID:
1. Log out from PocketOption
2. Log back in
3. Go to Settings → API → Regenerate SSID
4. Copy the new one
5. Update in bot settings
```

---

### ❌ Error 4: "Not enough data to train"

**Cause**: Date range too short or symbol inactive on those dates

**Fix**:
```
1. Increase date range: Start day 1, End day 30
2. Try different symbol: EURUSD_otc
3. Check if date is weekend (no trading)
```

---

### ❌ Error 5: "WARNING: Your GPU is not available"

**Cause**: CUDA/GPU not installed (optional)

**Fix**: 
This is **NOT critical**. Bot runs fine on CPU. To use GPU (faster):
```bash
# Install CUDA-compatible TensorFlow:
pip install tensorflow[and-cuda]
```

---

### ❌ Error 6: "KeyboardInterrupt"

**Cause**: You pressed Ctrl+C while bot was running

**Fix**:
```
This is normal. The bot gracefully stops.
Simply run it again:
python viper_ml_bot.py
```

---

### ❌ Error 7: "Backtest shows 0 trades"

**Cause**: 
- Date range has no active market hours
- Symbol not trading on those dates
- Confidence threshold too high (0.95+)

**Fix**:
```
1. Lower confidence threshold to 0.55
2. Try different dates or symbols
3. Check market is open (Mon-Fri 0:00-23:55 UTC)
```

---

### ❌ Error 8: "Win Rate 0% - All Losses"

**Cause**: 
- Model poorly trained (rare)
- Strategy doesn't fit current market
- Bug in feature extraction

**Fix**:
```
1. Retrain with more data: Extend date range
2. Try different symbol or timeframe
3. Check logs: tail -f viper_bot.log
4. Reset and restart
```

---

## 🎯 Advanced Usage

### Custom Feature Engineering

Edit `MultiTimeframeFeatureExtractor.extract()`:

```python
# Add your own feature
features['my_custom_feature'] = your_calculation

# Example: RSI divergence
features['rsi_divergence'] = current_rsi - prev_rsi
```

### Adjusting Model Hyperparameters

In `ModelEnsemble.__init__()`:

```python
# More aggressive model (higher overfitting risk)
self.xgb = XGBClassifier(
    n_estimators=200,      # More trees
    max_depth=8,           # Deeper trees
    learning_rate=0.2      # Faster learning
)

# More conservative (slower, safer)
self.xgb = XGBClassifier(
    n_estimators=50,       # Fewer trees
    max_depth=3,           # Shallower trees
    learning_rate=0.05     # Slower learning
)
```

### Online Learning Frequency

In `AdaptiveTrader.record_trade()`:

```python
# Retrain every 100 trades instead of 50
if self.trades_since_retrain >= 100:
    self.online_retrain()
```

### Dynamic Risk Sizing

In `DynamicRiskManager.calculate_position()`:

```python
# More aggressive (higher risk)
return min(balance*0.05, max(base_amount*2, vol_adj))

# More conservative (lower risk)
return min(balance*0.01, max(base_amount*0.5, vol_adj))
```

---

## 🏗️ Architecture

```
┌─ Data Pipeline
│  ├─ Download multi-TF candles (1m, 5m, 15m)
│  ├─ Store in memory (candles_dict)
│  └─ Pass to feature extractor
│
├─ Feature Engineering (100+ features)
│  ├─ Trend: EMA9/21/50, trend_up
│  ├─ Momentum: RSI14, MACD
│  ├─ Volatility: ATR, Bollinger Bands
│  └─ Price Action: candle patterns
│
├─ Model Ensemble
│  ├─ XGBoost (captures non-linear patterns)
│  ├─ CatBoost (handles categorical features)
│  ├─ Random Forest (ensemble voting)
│  └─ LSTM (temporal dependencies)
│
├─ Prediction & Meta-Model
│  ├─ Average probabilities from 4 models
│  ├─ Output: Signal (CALL/PUT) + Confidence %
│  └─ Filter by confidence threshold
│
├─ Risk Management
│  ├─ Kelly Criterion position sizing
│  ├─ Volatility adjustment (ATR-based)
│  └─ Loss streak detection
│
└─ Online Learning & Feedback
   ├─ Record trade result
   ├─ Update win rate
   └─ Retrain every 50 trades
```

---

## ❓ FAQ

### Q: Can I use this on real PocketOption account?

**A:** Yes, with `Mode = real`. But **start small** ($1-2 per trade) and **test in demo first** (at least 100 trades). Never risk money you can't afford to lose.

---

### Q: How often should I retrain the model?

**A:** The bot auto-retrains every 50 trades (Online Learning). For manual retraining:
```bash
# Stop the bot (Ctrl+C)
# Clear old model: rm -f model.pkl
# Run again: python viper_ml_bot.py
```

---

### Q: What's the expected win rate?

**A:** 
- Realistic: 55-65% on demo
- Very good: 65-75%
- Excellent: 75%+
- Impossible: 95%+ (likely overfitted)

Note: Binary options payouts mean you need ~56% winrate just to break even after commissions.

---

### Q: Can I run multiple symbols simultaneously?

**A:** Not in current version. To run multiple:
```bash
# Terminal 1
python viper_ml_bot.py  # GBPJPY_otc

# Terminal 2 (in separate window)
python viper_ml_bot.py  # EURUSD_otc
```

---

### Q: How do I integrate with Telegram alerts?

**A:** Edit `main()` to add:
```python
import requests

def send_telegram(message):
    requests.get(f'https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}')

# In backtest loop:
if win:
    send_telegram(f"✅ WIN: {signal} @ {price:.5f}")
```

---

### Q: What if my balance goes negative?

**A:** The bot stops trading when:
- `balance <= -max_loss` (e.g., -$20)
- `loss_streak >= 5` (5 consecutive losses)

Simply reconfigure and restart.

---

### Q: Is this guaranteed to make money?

**A:** **NO.** No trading bot is. Markets are unpredictable. This bot improves odds through ML, but:
- Past performance ≠ future results
- No strategy works in all market conditions
- You can lose money
- Use only what you can afford to lose

---

## 📝 License & Disclaimer

**This bot is for educational purposes.** Use at your own risk. The author is NOT responsible for financial losses. Always backtest thoroughly before deploying real capital.

---

## 💡 Tips for Success

1. **Start in Demo**: Test at least 1 week in demo mode
2. **Backtest First**: Always backtest date range before trading
3. **Small Positions**: Start with $1-2 per trade
4. **Daily Targets**: Set realistic targets ($5-20/day)
5. **Monitor Logs**: Check `viper_bot.log` for issues
6. **Adapt Strategy**: Adjust confidence, features based on market
7. **Never Martingale**: Don't double down on losses
8. **Take Breaks**: If 5 losses in a row, stop and analyze

---

## 📞 Support & Troubleshooting

**Check logs first:**
```bash
tail -f viper_bot.log
```

**Common issues with solutions in this README:**
- See section: [Common Errors & Fixes](#common-errors--fixes)

**Still stuck?** Review the Architecture section or re-read Configuration.

---

**Happy trading! 🚀**

Last updated: October 30, 2025
