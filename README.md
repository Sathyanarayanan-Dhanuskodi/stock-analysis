# Indian Stock Market Predictor

AI-powered stock prediction app for the Indian market (NSE/BSE) with swing trading, scalping, fundamentals analysis, and sector comparison tools.

## Features

- **Portfolio Watchlist** — Track stocks with buy price, quantity, and live P&L
- **Stock Screener** — Scan 25+ stocks for swing trading opportunities ranked by signal strength
- **Interactive Charts** — Candlestick charts with moving averages, Bollinger Bands, and volume
- **AI Predictions** — LSTM + XGBoost ensemble model for 1-30 day price forecasts
- **Swing Trading** — Support/resistance, Fibonacci, pivot points, pattern detection, and trade setups
- **Scalping** — 5-minute candle analysis with VWAP, fast EMAs, and Camarilla pivots
- **Fundamentals** — P/E, ROE, debt ratios, dividends, growth metrics with beginner-friendly explanations
- **Sector Comparison** — Normalized performance charts, correlation heatmaps, and peer metrics
- **News Sentiment** — AI-powered news analysis using Claude (requires Anthropic API key)

## Tech Stack

- **Frontend**: Streamlit
- **ML Models**: TensorFlow/Keras (LSTM), XGBoost
- **Data**: yfinance (NSE/BSE stocks)
- **Charts**: Plotly
- **Sentiment**: Anthropic Claude API

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd stock-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Add API key for sentiment analysis
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project Structure

```
stock-predictor/
├── app.py                  # Main Streamlit app (UI + all 9 tabs)
├── src/
│   ├── data_fetcher.py     # Stock data, fundamentals, screener
│   ├── feature_engineering.py  # 30+ technical indicators
│   ├── model.py            # LSTM model architecture
│   ├── xgboost_model.py    # XGBoost model
│   ├── ensemble.py         # LSTM + XGBoost blending
│   ├── predictor.py        # Training orchestration
│   ├── swing_trading.py    # Swing signals, S/R, Fibonacci, patterns
│   ├── scalping.py         # Intraday scalping signals
│   ├── hedge_trading.py    # Correlation, beta, Sharpe, VaR
│   └── sentiment.py        # Claude-powered news sentiment
├── .streamlit/config.toml  # Theme configuration
├── requirements.txt
├── .env.example
└── .gitignore
```

## Supported Stocks

25 popular Indian stocks (Reliance, TCS, HDFC Bank, Infosys, etc.) plus custom ticker input. Sector groupings: IT, Banking, FMCG, Auto, Pharma, Energy, Metals, Infrastructure.

## Disclaimer

Educational purposes only. Not financial advice. Always do your own research before making investment decisions.
