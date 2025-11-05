# 📁 Viper ML Bot - Project Structure & Files

## File Organization

```
binary-bot/
├── viper_ml_bot.py          ← MAIN BOT FILE (run this)
├── requirements.txt          ← All dependencies
├── README.md                 ← Full documentation
├── setup.sh                  ← Linux/macOS setup
├── setup.bat                 ← Windows setup
├── .venv/                    ← Virtual environment (created after setup)
├── viper_bot.log             ← Log file (auto-created)
└── models/                   ← Trained models (auto-created)
    ├── ensemble_xgb.pkl
    ├── ensemble_cat.pkl
    └── scaler.pkl
```

## Setup Instructions by OS

### 🐧 Linux / macOS

```bash
# 1. Clone/download the project
cd binary-bot

# 2. Run setup script (ONE TIME ONLY)
bash setup.sh

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Run the bot
python viper_ml_bot.py
```

### 🪟 Windows

```powershell
# 1. Open Command Prompt (cmd.exe)
# 2. Navigate to project folder
cd C:\path\to\binary-bot

# 3. Run setup script (ONE TIME ONLY)
setup.bat

# 4. Activate virtual environment
.venv\Scripts\activate

# 5. Run the bot
python viper_ml_bot.py
```

## Step-by-Step First Run

### Prerequisites
- ✅ Python 3.8+ installed ([download](https://www.python.org/downloads/))
- ✅ Stable internet connection
- ✅ PocketOption account
- ✅ Your SSID from PocketOption

### Installation (5 minutes)

```bash
# 1. Create a folder
mkdir binary-bot
cd binary-bot

# 2. Save these files in the folder:
#    - viper_ml_bot.py
#    - requirements.txt
#    - setup.sh (or setup.bat for Windows)
#    - README.md

# 3. Run setup (installs all dependencies)
bash setup.sh              # macOS/Linux
# OR
setup.bat                  # Windows

# 4. Verify installation
python -c "import xgboost; import tensorflow; print('✓ Ready')"
```

### First Backtest (10 minutes)

```bash
# 1. Activate environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# 2. Run bot
python viper_ml_bot.py

# 3. You'll see menu:
# ============================================================
# 🐍 VIPER ML TRADING BOT v3.0
# ============================================================
# 1. Configure Settings
# 2. Backtest
# 3. Show Configuration
# 4. Exit
# ============================================================

# 4. Select option 1 (Configure Settings)
# Select option: 1

# 5. Fill in your settings:
Symbol: GBPJPY_otc
SSID: [paste your SSID here]
Mode: d  (demo = safe testing)
Start day: 16
End day: 17
Target profit: 100
Max loss: 50
Base amount: 2
Confidence: 0.55

# 6. Select option 2 (Backtest)
# Select option: 2

# 7. Watch the backtest run...
# 📥 Downloading 3 TFs over 2 days...
# 🧠 Training ensemble on 1200 samples...
# 📊 BACKTEST RESULTS
# ... (trades will print)

# 8. Check results at the end
# ✓ Final: 45 trades, 28W/17L, WR: 62.2%
```

## Troubleshooting

### "Python not found"
```bash
# Check if Python is installed
python --version

# If not, download from:
https://www.python.org/downloads/
```

### "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### "SSID invalid"
```
1. Log out from PocketOption
2. Log back in
3. Settings → API → Regenerate SSID
4. Copy the new SSID
5. Paste into bot when prompted
```

### "No trades in backtest"
```
1. Lower confidence threshold: 0.55
2. Try different date range: 1-30
3. Try EURUSD_otc instead of GBPJPY_otc
```

## Useful Commands

```bash
# Activate virtual environment
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# Deactivate virtual environment
deactivate

# View logs while running
tail -f viper_bot.log          # Linux/macOS
type viper_bot.log             # Windows

# Clear logs
rm viper_bot.log               # Linux/macOS
del viper_bot.log              # Windows

# Remove virtual environment (if needed)
rm -rf .venv                   # Linux/macOS
rmdir /s /q .venv              # Windows

# Full reinstall
rm -rf .venv                   # Remove old env
bash setup.sh                  # Create new env
```

## File Descriptions

### viper_ml_bot.py
- **Size**: ~500 lines
- **Purpose**: Main bot executable
- **Contains**: 
  - `Config`: Configuration management
  - `MultiTimeframeFeatureExtractor`: Feature engineering
  - `ModelEnsemble`: ML models (XGBoost, CatBoost, LSTM)
  - `AdaptiveTrader`: Online learning
  - `DynamicRiskManager`: Risk management
  - `ViperMLBot`: Main trading bot
  - `main()`: Interactive CLI

### requirements.txt
- **Size**: 50 lines
- **Purpose**: List all Python packages to install
- **Install**: `pip install -r requirements.txt`

### README.md
- **Size**: 1000+ lines
- **Purpose**: Full documentation
- **Contains**: Features, setup, usage, FAQ, troubleshooting

### setup.sh / setup.bat
- **Purpose**: Automated setup for Linux/macOS and Windows
- **Does**: Creates venv, installs dependencies, verifies installation
- **Run once**: `bash setup.sh` or `setup.bat`

## First Time Tips

1. **Always use virtual environment**: Keeps dependencies isolated
2. **Test in DEMO first**: Never risk real money on first run
3. **Start small backtest**: Test 2-3 days before full month
4. **Check logs**: `viper_bot.log` has detailed error messages
5. **Read errors carefully**: They tell you exactly what's wrong
6. **Don't skip README**: It has answers to 80% of common issues

## Running in Background (Linux/macOS)

```bash
# Run bot and keep it going even if terminal closes
nohup python viper_ml_bot.py > output.log 2>&1 &

# View process
ps aux | grep viper

# Kill process
kill [PID]
```

## Running in Background (Windows)

```powershell
# Install Task Scheduler task
# Or use Python:
import subprocess
subprocess.Popen(['python', 'viper_ml_bot.py'])
```

## Next Steps After Setup

1. ✅ Run first backtest in DEMO
2. ✅ Understand the results (win rate, P/L)
3. ✅ Adjust confidence threshold if needed
4. ✅ Backtest different symbols (EURUSD_otc, USDCHF_otc)
5. ✅ Backtest longer date ranges (full month)
6. ✅ If >55% win rate consistently → consider real trading
7. ✅ If real trading → START SMALL ($1-2 per trade)

## Getting Help

1. **Check README.md** → Section: "Common Errors & Fixes"
2. **Check logs**: `tail -f viper_bot.log`
3. **Re-read Configuration Guide**
4. **Try demo backtest** on different dates/symbols
5. **Reinstall**: Remove .venv and run setup.sh again

Good luck! 🚀
