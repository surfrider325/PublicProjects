# S&P 500 Pattern Scanner

LSTM-powered scanner that detects 8 technical formations across the full S&P 500.

## Quick start

Double-click `launch_scanner.bat` â€” or run manually:

```bash
cd C:\Users\rcsch\Documents\python
streamlit run sp500_scanner_app.py
```

Then open http://localhost:8501 in your browser.

## How it works

1. The app loads the trained BiLSTM checkpoint (`sp500_pattern_lstm.pt`)
2. For each stock it downloads the latest 3 months of OHLCV data via yfinance
3. The model runs inference on the **most recent 30-day window**
4. Any pattern where confidence exceeds the tuned threshold is flagged as a signal

## Patterns detected

| # | Pattern | Sentiment |
|---|---------|-----------|
| 0 | Head & Shoulders | Bearish |
| 1 | Inv Head & Shoulders | Bullish |
| 2 | Double Top | Bearish |
| 3 | Double Bottom | Bullish |
| 4 | Rising Wedge | Bearish |
| 5 | Falling Wedge | Bullish |
| 6 | Bull Flag | Bullish |
| 7 | Bear Flag | Bearish |

## Settings (sidebar)

- **Checkpoint path** â€” point to `sp500_pattern_lstm.pt`
- **Sentiment filter** â€” All / Bullish only / Bearish only
- **Min confidence** â€” slide to raise the detection bar (70%+ recommended)
- **Pattern types** â€” deselect patterns to hide
- **Scan scope** â€” full S&P 500 (503 stocks, ~2-3 min) or a custom ticker list

## Performance notes

- GPU (RTX 5070): full S&P 500 scan takes ~2-3 minutes
- CPU only: ~8-12 minutes
- Price data is cached for 15 min; model weights cached until restart
- Click **Run scan** to refresh

## Model details

- Architecture: BiLSTM (192 units x 3 layers, bidirectional) + temporal attention
- Parameters: 2.3M
- Training: 501 S&P 500 stocks, 80k windows, 60 epochs, focal loss gamma=2.5
- Best macro F1: 35.7% (test set) | Weighted F1: 38.7%
- Best classes: Double Bottom 46.7%, Double Top 42.0%, Bull Flag 39.0%

## Disclaimer

This tool is for educational and research purposes only. Not financial advice.
