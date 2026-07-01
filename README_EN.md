# 📈 Stock Analyzer

A **Streamlit** web app for **technical + fundamental analysis** of stocks, ETFs,
commodities, crypto, bonds and indices. Enter a symbol, press **Analyze**, and get
charts, indicators, fundamentals, recent news, a macro backdrop, and an automatic
trend assessment — all explained in plain language.

> ⚠️ **For educational purposes only. This is not investment advice.**

---

## What it does

**Markets & assets (via a simple dropdown — no need to memorize symbols)**
- Stocks: USA, Greece (ATHEX), Germany, France, Netherlands, London, Tokyo, Sydney
- ETFs / indices: SPY, VOO, QQQ, VUAA, VWCE, and raw indices (^GSPC, ^GDAXI, GD.AT…)
- Commodities: gold, silver, oil (WTI/Brent), natural gas, copper, wheat, corn, coffee…
- Crypto: BTC, ETH, SOL, XRP, and more
- Bonds: US Treasury yields (^TNX etc.) and bond ETFs (TLT, IEF, AGG, LQD, HYG…)
- Anything else: an "Other" option accepts **any** Yahoo Finance symbol worldwide

**Analysis (uses 14+ indicators under the hood)**
- Technical: RSI, ATR, MACD, Bollinger Bands, SMA50/200, MA20/EMA20, Stochastic,
  ADX, OBV, CCI, Williams %R, ROC, MFI, CMF
- **Trend score** from −100 to +100 combining all signals, shown as a verdict
- **Buy / Hold / Sell** pie chart derived purely from the analysis (no opinion)
- Plain-language explanations of the **5 most relevant** indicators for each asset
- Fundamentals (for companies): Market Cap, P/E, EPS, revenue, Free/Operating
  Cash Flow, dividend yield, beta, analyst mean target, cash-flow table
- **News** tab: recent headlines per symbol with source and links
- **Macro & Events** tab: 10Y yield, VIX, Dollar Index + upcoming high-impact events
  (FOMC, ECB) with countdowns and neutral "what to watch" explanations

**Chart**
- Candlestick / line toggle, plus Bollinger, SMA overlays, volume, RSI and MACD panels

---

## Install (once)

You need **Python 3.9+**. In a terminal:

```bash
cd stock_analyzer
pip install -r requirements.txt
```

## Run

```bash
streamlit run app_en.py
```

It opens automatically in your browser (usually http://localhost:8501).

> On Windows PowerShell, if `streamlit` isn't recognized, use:
> `python -m streamlit run app_en.py`
> (and make sure you are inside the `stock_analyzer` folder first — see INSTRUCTIONS_EN.md)

---

## How to use

1. Pick a **market** in the left panel (e.g. "USA", "Greece", "Commodities").
2. Type a symbol (or pick from the list). The correct exchange suffix is added
   automatically — e.g. choose "Greece" + type `ETE` → `ETE.AT`.
3. Choose a **time period** (1 month … max).
4. Press **🔍 Analyze**.
5. Explore the tabs: **Chart · Technical signals · Fundamentals · News · Macro & Events**.

**Symbol tips**
- Tokyo symbols are **numbers**: `7203.T` (Toyota), `6758.T` (Sony)
- Greece uses the **.AT** suffix: `ETE.AT`, `OPAP.AT`
- European ETFs vary by exchange: `VUAA.DE` (€) or `VUAA.L` (London)

---

## Notes

- **Data source:** Yahoo Finance (free, no API key required).
- **Why not FRED?** FRED has macro series (rates, inflation) but **no individual
  stock data**, so the app uses Yahoo for prices/fundamentals and market gauges
  (10Y yield, VIX, DXY) for the macro backdrop.
- **Rate limits:** Yahoo may temporarily block rapid requests. The app retries
  automatically; if it still fails, wait ~30–60s and use the "Clear cache & retry"
  button.
- The 2025–2026 central-bank dates are official; 2027 dates are indicative
  (published ~1 year ahead).
- Nobody can predict how an event will move a price — outcomes depend on whether
  results beat or miss the market's forecast. The app explains **what to watch**,
  not what will happen.

---

## Files

- `app.py` — Greek version
- `app_en.py` — English version (this one)
- `requirements.txt` — dependencies
- `README_EN.md` — this file
- `INSTRUCTIONS_EN.md` — detailed step-by-step setup & deploy guide
