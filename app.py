"""
Stock Analyzer — Τεχνική + Θεμελιώδης Ανάλυση Μετοχών
=====================================================
Streamlit web app. Γράφεις σύμβολο μετοχής (π.χ. AAPL, MSFT, TSLA),
και βλέπεις: τιμή, RSI, ATR, MACD, Bollinger Bands, SMA50/200, MA,
θεμελιώδη (market cap, cash flow κλπ), μακρο-υπόβαθρο (επιτόκια Fed),
ημερολόγιο FOMC/ΕΚΤ, και αυτόματη εκτίμηση τάσης.

Εκτέλεση:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# ΡΥΘΜΙΣΕΙΣ ΣΕΛΙΔΑΣ
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# ΤΕΧΝΙΚΟΙ ΔΕΙΚΤΕΣ (καθαρά με pandas/numpy, χωρίς εξωτερικές βιβλιοθήκες TA)
# ----------------------------------------------------------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — μέτρο μεταβλητότητας."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD line, signal line, histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands (mid, upper, lower)."""
    mid = sma(series, window)
    std = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει όλους τους δείκτες σε ένα DataFrame OHLCV."""
    out = df.copy()
    close = out["Close"]
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["MA20"] = sma(close, 20)
    out["RSI"] = rsi(close, 14)
    out["ATR"] = atr(out, 14)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(close)
    out["BB_mid"], out["BB_up"], out["BB_low"] = bollinger(close)
    return out


# ----------------------------------------------------------------------------
# ΑΝΑΓΝΩΡΙΣΗ ΤΑΣΗΣ (συνδυασμός σημάτων -> σκορ)
# ----------------------------------------------------------------------------
def trend_analysis(df: pd.DataFrame) -> dict:
    """Επιστρέφει σκορ τάσης (-100 έως +100) και επιμέρους σήματα."""
    last = df.iloc[-1]
    signals = []
    score = 0

    # 1. Θέση τιμής vs SMA200 (μακροπρόθεσμη τάση)
    if not np.isnan(last["SMA200"]):
        if last["Close"] > last["SMA200"]:
            score += 25; signals.append(("Τιμή > SMA200", "bullish", "Μακροπρόθεσμα ανοδικά"))
        else:
            score -= 25; signals.append(("Τιμή < SMA200", "bearish", "Μακροπρόθεσμα καθοδικά"))

    # 2. SMA50 vs SMA200 (golden/death cross)
    if not np.isnan(last["SMA50"]) and not np.isnan(last["SMA200"]):
        if last["SMA50"] > last["SMA200"]:
            score += 20; signals.append(("SMA50 > SMA200", "bullish", "Golden cross — ανοδικό μομέντουμ"))
        else:
            score -= 20; signals.append(("SMA50 < SMA200", "bearish", "Death cross — καθοδικό μομέντουμ"))

    # 3. RSI
    r = last["RSI"]
    if not np.isnan(r):
        if r > 70:
            score -= 10; signals.append((f"RSI {r:.0f}", "bearish", "Υπεραγορασμένη — πιθανή διόρθωση"))
        elif r < 30:
            score += 10; signals.append((f"RSI {r:.0f}", "bullish", "Υπερπουλημένη — πιθανή ανάκαμψη"))
        elif r > 50:
            score += 5; signals.append((f"RSI {r:.0f}", "bullish", "Μομέντουμ θετικό"))
        else:
            score -= 5; signals.append((f"RSI {r:.0f}", "bearish", "Μομέντουμ αρνητικό"))

    # 4. MACD
    if not np.isnan(last["MACD"]) and not np.isnan(last["MACD_signal"]):
        if last["MACD"] > last["MACD_signal"]:
            score += 15; signals.append(("MACD > Signal", "bullish", "Ανοδικό σήμα MACD"))
        else:
            score -= 15; signals.append(("MACD < Signal", "bearish", "Καθοδικό σήμα MACD"))

    # 5. Bollinger position
    if not np.isnan(last["BB_up"]) and not np.isnan(last["BB_low"]):
        if last["Close"] > last["BB_up"]:
            score -= 5; signals.append(("Πάνω από BB", "bearish", "Πιθανή υπερβολή"))
        elif last["Close"] < last["BB_low"]:
            score += 5; signals.append(("Κάτω από BB", "bullish", "Πιθανό oversold"))

    score = max(-100, min(100, score))
    if score >= 40:
        verdict, color = "ΙΣΧΥΡΑ ΑΝΟΔΙΚΗ", "#1a9850"
    elif score >= 10:
        verdict, color = "ΗΠΙΑ ΑΝΟΔΙΚΗ", "#66bd63"
    elif score > -10:
        verdict, color = "ΟΥΔΕΤΕΡΗ / ΠΛΑΓΙΑ", "#999999"
    elif score > -40:
        verdict, color = "ΗΠΙΑ ΚΑΘΟΔΙΚΗ", "#f46d43"
    else:
        verdict, color = "ΙΣΧΥΡΑ ΚΑΘΟΔΙΚΗ", "#d73027"

    return {"score": score, "verdict": verdict, "color": color, "signals": signals}


def simple_explanation(trend: dict, last) -> str:
    """Εξήγηση 2-3 γραμμών σε πολύ απλά λόγια (σαν σε παιδί)."""
    score = trend["score"]
    rsi_val = last["RSI"]

    # Βασική «ιστορία» ανάλογα με το σκορ
    if score >= 40:
        story = ("Η μετοχή ανεβαίνει σταθερά τον τελευταίο καιρό και δείχνει δύναμη — "
                 "σαν παιδί που τρέχει και έχει ακόμα πολλή ενέργεια.")
    elif score >= 10:
        story = ("Η μετοχή ανεβαίνει ήπια, αλλά όχι πολύ δυνατά — "
                 "σαν περπάτημα ανηφόρα με αργό ρυθμό.")
    elif score > -10:
        story = ("Η μετοχή ούτε ανεβαίνει ούτε κατεβαίνει ξεκάθαρα — "
                 "κινείται στο ίδιο σημείο, σαν να ξεκουράζεται.")
    elif score > -40:
        story = ("Η μετοχή κατεβαίνει ήπια — "
                 "σαν μπάλα που κυλάει αργά προς τα κάτω.")
    else:
        story = ("Η μετοχή πέφτει αρκετά τον τελευταίο καιρό και δείχνει αδυναμία — "
                 "σαν μπάλα που κατρακυλάει γρήγορα στον κατήφορο.")

    # Σχόλιο RSI σε απλά λόγια
    if not np.isnan(rsi_val):
        if rsi_val > 70:
            rsi_note = " Έχει ανέβει πολύ γρήγορα, οπότε ίσως χρειαστεί μια μικρή ξεκούραση (πτώση)."
        elif rsi_val < 30:
            rsi_note = " Έχει πέσει πολύ, οπότε ίσως είναι κουρασμένη και θελήσει να ανέβει λίγο."
        else:
            rsi_note = ""
    else:
        rsi_note = ""

    return story + rsi_note


def explain_indicators(last, df) -> list:
    """Επιστρέφει λίστα από (τίτλος, τιμή, εξήγηση σε απλά λόγια) για κάθε δείκτη."""
    items = []
    close = last["Close"]

    # --- RSI ---
    r = last["RSI"]
    if np.isnan(r):
        rsi_txt = "Δεν υπάρχουν αρκετά δεδομένα ακόμα."
    elif r > 70:
        rsi_txt = (f"Το RSI είναι {r:.0f}, δηλαδή πάνω από 70. Αυτό λέγεται «υπεραγορασμένη»: "
                   "η μετοχή ανέβηκε πολύ γρήγορα και ίσως χρειαστεί ξεκούραση ή μικρή πτώση. "
                   "Σκέψου το σαν δρομέα που έτρεξε πολύ δυνατά και λαχανιάζει.")
    elif r < 30:
        rsi_txt = (f"Το RSI είναι {r:.0f}, δηλαδή κάτω από 30. Αυτό λέγεται «υπερπουλημένη»: "
                   "η μετοχή έπεσε πολύ και ίσως είναι έτοιμη να ανακάμψει. "
                   "Σαν ελατήριο που πιέστηκε πολύ και θέλει να πεταχτεί πάνω.")
    elif r >= 50:
        rsi_txt = (f"Το RSI είναι {r:.0f}, δηλαδή λίγο πάνω από τη μέση (50). "
                   "Δείχνει ότι οι αγοραστές έχουν ένα ελαφρύ προβάδισμα — υγιές, χωρίς υπερβολές.")
    else:
        rsi_txt = (f"Το RSI είναι {r:.0f}, δηλαδή λίγο κάτω από τη μέση (50). "
                   "Δείχνει ότι οι πωλητές έχουν ένα ελαφρύ προβάδισμα, αλλά τίποτα ακραίο.")
    items.append(("RSI (δείκτης δύναμης)", f"{r:.0f}" if not np.isnan(r) else "—", rsi_txt))

    # --- SMA50 vs SMA200 ---
    s50, s200 = last["SMA50"], last["SMA200"]
    if np.isnan(s50) or np.isnan(s200):
        sma_txt = ("Χρειάζονται περισσότερα δεδομένα για τους μέσους όρους (ο SMA200 θέλει "
                   "200 ημέρες). Διάλεξε μεγαλύτερο χρονικό διάστημα (π.χ. 1 έτος) για να εμφανιστούν.")
        sma_val = "—"
    elif s50 > s200:
        sma_txt = (f"Ο μέσος όρος 50 ημερών ({s50:.2f}) είναι πάνω από τον μέσο όρο 200 ημερών "
                   f"({s200:.2f}). Αυτό λέγεται «golden cross» και θεωρείται καλό σημάδι: "
                   "η πρόσφατη πορεία είναι καλύτερη από τη μακροχρόνια, δηλαδή ανοδική τάση.")
        sma_val = "Ανοδική"
    else:
        sma_txt = (f"Ο μέσος όρος 50 ημερών ({s50:.2f}) είναι κάτω από τον μέσο όρο 200 ημερών "
                   f"({s200:.2f}). Αυτό λέγεται «death cross» και θεωρείται προειδοποίηση: "
                   "η πρόσφατη πορεία είναι χειρότερη από τη μακροχρόνια, δηλαδή καθοδική τάση.")
        sma_val = "Καθοδική"
    items.append(("Μέσοι όροι (SMA50 vs SMA200)", sma_val, sma_txt))

    # --- Τιμή vs SMA200 ---
    if not np.isnan(s200):
        if close > s200:
            p200_txt = (f"Η τιμή ({close:.2f}) είναι πάνω από τον μέσο όρο 200 ημερών ({s200:.2f}). "
                        "Σε γενικές γραμμές, όταν η τιμή είναι πάνω από αυτή τη γραμμή, η μετοχή "
                        "θεωρείται ότι βρίσκεται σε μακροπρόθεσμα ανοδικό «κλίμα».")
        else:
            p200_txt = (f"Η τιμή ({close:.2f}) είναι κάτω από τον μέσο όρο 200 ημερών ({s200:.2f}). "
                        "Όταν η τιμή είναι κάτω από αυτή τη γραμμή, η μετοχή θεωρείται ότι βρίσκεται "
                        "σε μακροπρόθεσμα καθοδικό «κλίμα».")
        items.append(("Τιμή σε σχέση με SMA200", "Πάνω" if close > s200 else "Κάτω", p200_txt))

    # --- MACD ---
    m, sig = last["MACD"], last["MACD_signal"]
    if np.isnan(m) or np.isnan(sig):
        macd_txt = "Δεν υπάρχουν αρκετά δεδομένα ακόμα."
        macd_val = "—"
    elif m > sig:
        macd_txt = ("Η γραμμή MACD είναι πάνω από τη γραμμή «signal». Αυτό είναι ανοδικό σήμα: "
                    "το μομέντουμ (η «φόρα») της μετοχής στρέφεται προς τα πάνω.")
        macd_val = "Ανοδικό"
    else:
        macd_txt = ("Η γραμμή MACD είναι κάτω από τη γραμμή «signal». Αυτό είναι καθοδικό σήμα: "
                    "η «φόρα» της μετοχής στρέφεται προς τα κάτω.")
        macd_val = "Καθοδικό"
    items.append(("MACD (φόρα/μομέντουμ)", macd_val, macd_txt))

    # --- Bollinger Bands ---
    bb_up, bb_low, bb_mid = last["BB_up"], last["BB_low"], last["BB_mid"]
    if np.isnan(bb_up) or np.isnan(bb_low):
        bb_txt = "Δεν υπάρχουν αρκετά δεδομένα ακόμα."
        bb_val = "—"
    elif close > bb_up:
        bb_txt = ("Η τιμή ξεπέρασε την πάνω «μπάντα» Bollinger. Συνήθως σημαίνει ότι ανέβηκε "
                  "πολύ απότομα και ίσως «επιστρέψει» λίγο προς τα κάτω για να ισορροπήσει.")
        bb_val = "Πάνω μπάντα"
    elif close < bb_low:
        bb_txt = ("Η τιμή έπεσε κάτω από την κάτω «μπάντα» Bollinger. Συνήθως σημαίνει ότι έπεσε "
                  "πολύ απότομα και ίσως «αναπηδήσει» λίγο προς τα πάνω.")
        bb_val = "Κάτω μπάντα"
    else:
        bb_txt = (f"Η τιμή κινείται ανάμεσα στις δύο μπάντες (γύρω από το μέσο {bb_mid:.2f}). "
                  "Αυτό είναι το «κανονικό» εύρος — καμία ακραία κίνηση αυτή τη στιγμή.")
        bb_val = "Εντός εύρους"
    items.append(("Bollinger Bands (εύρος τιμής)", bb_val, bb_txt))

    # --- ATR ---
    a = last["ATR"]
    if np.isnan(a):
        atr_txt = "Δεν υπάρχουν αρκετά δεδομένα ακόμα."
        atr_val = "—"
    else:
        pct = a / close * 100 if close else 0
        if pct > 4:
            level = "υψηλή"
            extra = "Η μετοχή κινείται έντονα — μεγαλύτερο ρίσκο αλλά και μεγαλύτερες ευκαιρίες."
        elif pct > 2:
            level = "μέτρια"
            extra = "Φυσιολογικές καθημερινές διακυμάνσεις."
        else:
            level = "χαμηλή"
            extra = "Η μετοχή κινείται ήρεμα, με μικρές καθημερινές αλλαγές."
        atr_txt = (f"Το ATR είναι {a:.2f}, δηλαδή περίπου {pct:.1f}% της τιμής. Δείχνει "
                   f"{level} μεταβλητότητα — δηλαδή πόσο «νευρικά» κινείται η τιμή μέσα στην ημέρα. {extra}")
        atr_val = f"{a:.2f}"
    items.append(("ATR (μεταβλητότητα)", atr_val, atr_txt))

    return items


def buy_hold_sell(score: int) -> dict:
    """Μετατρέπει το σκορ τάσης σε ποσοστά Buy / Hold / Sell (αθροίζουν 100)."""
    # Όσο πιο ανοδικό το σκορ, τόσο μεγαλύτερο το Buy· όσο πιο καθοδικό, τόσο το Sell.
    # Το Hold είναι μεγαλύτερο όταν το σκορ είναι κοντά στο μηδέν (αβεβαιότητα).
    s = max(-100, min(100, score))
    buy = max(0, s)               # 0..100
    sell = max(0, -s)             # 0..100
    hold = 100 - abs(s)           # μεγαλύτερο κοντά στο 0
    total = buy + hold + sell
    if total == 0:
        buy, hold, sell = 0, 100, 0
        total = 100
    return {
        "Buy": round(buy / total * 100),
        "Hold": round(hold / total * 100),
        "Sell": round(sell / total * 100),
    }


# ----------------------------------------------------------------------------
# FUNDAMENTAL — δεδομένα εταιρείας μέσω yfinance
# ----------------------------------------------------------------------------
def fmt_num(n):
    """Μορφοποίηση μεγάλων αριθμών (T/B/M)."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.2f}"


@st.cache_data(ttl=900, show_spinner=False)
def load_stock(ticker: str, period: str):
    """Κατεβάζει ιστορικό + θεμελιώδη, με αυτόματες επαναπροσπάθειες (retry)
    σε περίπτωση προσωρινού rate-limit του Yahoo. Cache 15 λεπτά."""
    import time

    hist = None
    last_err = None
    # Έως 3 προσπάθειες με αυξανόμενη αναμονή (1s, 2s, 4s)
    for attempt in range(3):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, auto_adjust=False)
            if hist is not None and not hist.empty:
                break  # επιτυχία
        except Exception as e:
            last_err = e
        time.sleep(2 ** attempt)  # 1, 2, 4 δευτερόλεπτα

    # Αν μετά τις προσπάθειες δεν έχουμε δεδομένα, επιστρέφουμε άδειο + λόγο
    if hist is None or hist.empty:
        return None, {}, None, "rate_limit_or_invalid"

    # Θεμελιώδη (δεν χαλάει η ροή αν αποτύχουν)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    cashflow = None
    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = None
    return hist, info, cashflow, "ok"


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro():
    """Μακρο-υπόβαθρο μέσω yfinance (χωρίς FRED key): 10Y yield, VIX, DXY."""
    out = {}
    for name, sym in [("10Y Treasury", "^TNX"), ("VIX (φόβος)", "^VIX"), ("Dollar Index", "DX-Y.NYB")]:
        try:
            h = yf.Ticker(sym).history(period="1mo")
            if not h.empty:
                cur = h["Close"].iloc[-1]
                prev = h["Close"].iloc[0]
                chg = (cur - prev) / prev * 100
                out[name] = (cur, chg)
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------
# ΗΜΕΡΟΛΟΓΙΟ FOMC / ΕΚΤ 2025-2026 (στατικό — επίσημες ημερομηνίες)
# ----------------------------------------------------------------------------
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]
ECB_DATES = [
    "2025-01-30", "2025-03-06", "2025-04-17", "2025-06-05",
    "2025-07-24", "2025-09-11", "2025-10-30", "2025-12-18",
    "2026-01-29", "2026-03-12", "2026-04-16", "2026-06-04",
    "2026-07-23", "2026-09-10", "2026-10-29", "2026-12-17",
]


def next_meeting(dates):
    today = dt.date.today()
    upcoming = [dt.date.fromisoformat(d) for d in dates if dt.date.fromisoformat(d) >= today]
    return min(upcoming) if upcoming else None


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #888; margin-top: 0; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📈 Stock Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Τεχνική + Θεμελιώδης ανάλυση μετοχών με μακρο-υπόβαθρο</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Ρυθμίσεις")

    # Αγορές: όνομα -> (κατάληξη Yahoo, παραδείγματα συμβόλων)
    MARKETS = {
        "🇺🇸 ΗΠΑ (NYSE/Nasdaq)": ("", ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]),
        "🇬🇷 Ελλάδα (Χ.Α.)":      (".AT", ["ETE.AT", "OPAP.AT", "AEGN.AT", "MYTIL.AT", "EUROB.AT"]),
        "🇩🇪 Γερμανία (Xetra)":   (".DE", ["BMW.DE", "SAP.DE", "VOW3.DE", "SIE.DE"]),
        "🇫🇷 Γαλλία (Euronext)":  (".PA", ["MC.PA", "AIR.PA", "OR.PA", "BNP.PA"]),
        "🇳🇱 Ολλανδία (Euronext)": (".AS", ["ASML.AS", "INGA.AS", "HEIA.AS"]),
        "🇬🇧 Λονδίνο (LSE)":      (".L", ["HSBA.L", "BP.L", "SHEL.L", "VOD.L"]),
        "🇯🇵 Τόκιο (TSE)":        (".T", ["7203.T", "6758.T", "9984.T", "8306.T"]),
        "🇦🇺 Σίδνεϊ (ASX)":       (".AX", ["BHP.AX", "CBA.AX", "CSL.AX", "WBC.AX"]),
        "🛢️ Commodities (εμπορεύματα)": ("LIST:COMMODITY", []),
        "📈 ETFs / Δείκτες (indices)": ("LIST:ETF", []),
        "₿ Crypto (κρυπτονομίσματα)": ("LIST:CRYPTO", []),
        "📜 Bonds (ομόλογα & ETFs)": ("LIST:BONDS", []),
        "✏️ Άλλο / γράφω μόνος μου": (None, []),
    }

    # Λίστες φιλικών ονομάτων -> σύμβολα Yahoo
    COMMODITIES = {
        "🥇 Χρυσός (Gold)": "GC=F",
        "🥈 Ασήμι (Silver)": "SI=F",
        "🛢️ Πετρέλαιο WTI (Crude Oil)": "CL=F",
        "🛢️ Πετρέλαιο Brent": "BZ=F",
        "🔥 Φυσικό αέριο (Natural Gas)": "NG=F",
        "🟤 Χαλκός (Copper)": "HG=F",
        "🪙 Πλατίνα (Platinum)": "PL=F",
        "🌾 Σιτάρι (Wheat)": "ZW=F",
        "🌽 Καλαμπόκι (Corn)": "ZC=F",
        "☕ Καφές (Coffee)": "KC=F",
    }
    CRYPTO = {
        "₿ Bitcoin": "BTC-USD",
        "Ξ Ethereum": "ETH-USD",
        "◎ Solana": "SOL-USD",
        "✕ XRP": "XRP-USD",
        "🐕 Dogecoin": "DOGE-USD",
        "🔷 Cardano": "ADA-USD",
        "🔺 Avalanche": "AVAX-USD",
        "🔗 Chainlink": "LINK-USD",
    }
    BONDS = {
        "🇺🇸 Απόδοση 10ετούς ΗΠΑ (yield)": "^TNX",
        "🇺🇸 Απόδοση 30ετούς ΗΠΑ (yield)": "^TYX",
        "🇺🇸 Απόδοση 5ετούς ΗΠΑ (yield)": "^FVX",
        "🇺🇸 Απόδοση 13 εβδ. ΗΠΑ (yield)": "^IRX",
        "📦 TLT — 20+ ετών ΗΠΑ (ETF)": "TLT",
        "📦 IEF — 7-10 ετών ΗΠΑ (ETF)": "IEF",
        "📦 SHY — 1-3 ετών ΗΠΑ (ETF)": "SHY",
        "📦 AGG — Σύνολο αγοράς ομολόγων (ETF)": "AGG",
        "📦 LQD — Εταιρικά ομόλογα (ETF)": "LQD",
        "📦 HYG — Ομόλογα υψηλής απόδοσης (ETF)": "HYG",
    }
    ETFS = {
        "🇺🇸 SPY — S&P 500 (ΗΠΑ)": "SPY",
        "🇺🇸 VOO — S&P 500 (Vanguard)": "VOO",
        "🇺🇸 IVV — S&P 500 (iShares)": "IVV",
        "🇺🇸 QQQ — Nasdaq 100 (τεχνολογία)": "QQQ",
        "🇺🇸 DIA — Dow Jones 30": "DIA",
        "🇺🇸 IWM — Russell 2000 (μικρές εταιρ.)": "IWM",
        "🇺🇸 VTI — Σύνολο αγοράς ΗΠΑ": "VTI",
        "🌍 VT — Όλος ο κόσμος (Vanguard)": "VT",
        "🌍 VWCE.DE — FTSE All-World (EU/€)": "VWCE.DE",
        "🇪🇺 VUAA.DE — S&P 500 (EU/€, acc)": "VUAA.DE",
        "🇬🇧 VUAA.L — S&P 500 (Λονδίνο)": "VUAA.L",
        "🇪🇺 CSPX.L — S&P 500 (iShares, €)": "CSPX.L",
        "🇪🇺 EUNL.DE — MSCI World (iShares)": "EUNL.DE",
        "🌏 EEM — Αναδυόμενες αγορές": "EEM",
        "🇪🇺 EZU — Ευρωζώνη": "EZU",
        "📊 ^GSPC — Δείκτης S&P 500 (raw)": "^GSPC",
        "📊 ^NDX — Δείκτης Nasdaq 100 (raw)": "^NDX",
        "📊 ^DJI — Δείκτης Dow Jones (raw)": "^DJI",
        "📊 ^GDAXI — Δείκτης DAX Γερμανίας": "^GDAXI",
        "📊 GD.AT — Γενικός Δείκτης Χ.Α.": "GD.AT",
    }
    PICK_LISTS = {
        "LIST:COMMODITY": ("εμπόρευμα", COMMODITIES,
                           "Τα εμπορεύματα δεν έχουν θεμελιώδη — μόνο τεχνική ανάλυση."),
        "LIST:ETF": ("ETF / δείκτη", ETFS,
                     "Τα ETFs «καλάθια» πολλών μετοχών. Τα ^ είναι δείκτες (indices). Δουλεύει πλήρως η τεχνική ανάλυση."),
        "LIST:CRYPTO": ("κρυπτονόμισμα", CRYPTO,
                        "Τα κρυπτονομίσματα δεν έχουν θεμελιώδη — μόνο τεχνική ανάλυση. Προσοχή: πολύ υψηλή μεταβλητότητα."),
        "LIST:BONDS": ("ομόλογο / ETF", BONDS,
                       "Τα yields (^) δείχνουν απόδοση· τα ETFs (📦) διαπραγματεύονται σαν μετοχές."),
    }

    market = st.selectbox("Αγορά / Χρηματιστήριο", list(MARKETS.keys()), index=0)
    suffix, examples = MARKETS[market]

    if suffix in PICK_LISTS:
        kind_label, pick_dict, note = PICK_LISTS[suffix]
        choice = st.selectbox(f"Διάλεξε {kind_label}", list(pick_dict.keys()), index=0)
        ticker = pick_dict[choice]
        st.caption(note)
    else:
        is_manual = suffix is None
        raw = st.text_input(
            "Σύμβολο" if is_manual else "Σύμβολο μετοχής",
            value=examples[0] if examples else "AAPL",
            help=("Γράψε οποιοδήποτε σύμβολο του Yahoo Finance: μετοχή, ETF, index, "
                  "futures — με τη σωστή κατάληξη αγοράς αν χρειάζεται."
                  if is_manual else
                  "Γράψε μόνο το σύμβολο — η κατάληξη της αγοράς προστίθεται αυτόματα."),
        ).strip().upper()

        # Αυτόματη προσθήκη κατάληξης (αν δεν την έχει ήδη γράψει ο χρήστης)
        if suffix and raw and not raw.endswith(suffix):
            # αν έγραψε ήδη κάποια άλλη κατάληξη (π.χ. .DE) σεβόμαστε αυτό που έγραψε
            if "." not in raw:
                ticker = raw + suffix
            else:
                ticker = raw
        else:
            ticker = raw

        if is_manual:
            st.caption("💡 Εδώ μπορείς να αναλύσεις **οτιδήποτε** υπάρχει στο Yahoo Finance — "
                       "οποιαδήποτε μετοχή, ETF ή δείκτη στον κόσμο.")

        if examples:
            st.caption("Δημοφιλή: " + " · ".join(examples))

    period = st.selectbox(
        "Χρονικό διάστημα",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3,
        format_func=lambda p: {
            "1mo": "1 μήνας", "3mo": "3 μήνες", "6mo": "6 μήνες",
            "1y": "1 έτος", "2y": "2 έτη", "5y": "5 έτη", "max": "Μέγιστο",
        }.get(p, p),
    )
    analyze = st.button("🔍 Ανάλυση", type="primary", use_container_width=True)
    if suffix:
        st.caption(f"🔎 Θα αναλυθεί: **{ticker}**")
    st.divider()
    st.caption("Πηγή τιμών: Yahoo Finance · Μακρο: TNX/VIX/DXY")
    st.caption("⚠️ Μόνο για εκπαιδευτικούς σκοπούς — όχι επενδυτική συμβουλή.")

if analyze or ticker:
    try:
        with st.spinner(f"Φόρτωση δεδομένων για {ticker}… (με αυτόματες επαναπροσπάθειες)"):
            hist, info, cashflow, status = load_stock(ticker, period)

        if hist is None or hist.empty:
            st.error(
                f"⚠️ Δεν βρέθηκαν δεδομένα για «{ticker}» αυτή τη στιγμή.\n\n"
                "Αν το σύμβολο είναι σωστό (π.χ. NVDA, AAPL), το πιο πιθανό είναι ότι το "
                "**Yahoo Finance σε μπλόκαρε προσωρινά** επειδή έγιναν πολλά αιτήματα γρήγορα "
                "(rate-limit). Δεν φταις εσύ ούτε το σύμβολο."
            )
            cretry, cinfo = st.columns([1, 2])
            with cretry:
                if st.button("🔄 Καθάρισε & ξαναδοκίμασε"):
                    st.cache_data.clear()
                    st.rerun()
            with cinfo:
                st.caption(
                    "Ή περίμενε 30-60 δευτερόλεπτα και πάτα ξανά Ανάλυση. "
                    "Αν επιμένει, κλείσε την εφαρμογή (Ctrl+C) και ξανάνοιξέ την."
                )
            st.divider()
            st.caption(
                "Έλεγξε επίσης: στο 🇯🇵 Τόκιο τα σύμβολα είναι **αριθμοί** (7203.T = Toyota) · "
                "στην 🇬🇷 Ελλάδα η κατάληξη είναι **.AT** (ETE.AT)."
            )
            st.stop()

        df = compute_indicators(hist)
        last = df.iloc[-1]
        trend = trend_analysis(df)
        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")

        # --- Κεφαλίδα: τιμή + τάση ---
        price = last["Close"]
        prev = df["Close"].iloc[-2] if len(df) > 1 else price
        chg = (price - prev) / prev * 100

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1.4])
        c1.markdown(f"### {name}")
        c1.caption(f"{ticker} · {info.get('sector', '')} {info.get('industry', '')}")
        c2.metric("Τιμή", f"{price:.2f} {currency}", f"{chg:+.2f}%")
        c3.metric("RSI (14)", f"{last['RSI']:.0f}" if not np.isnan(last['RSI']) else "—")
        c4.markdown(
            f"<div style='padding:8px 14px;border-radius:10px;background:{trend['color']}22;"
            f"border:1px solid {trend['color']};text-align:center'>"
            f"<div style='font-size:0.75rem;color:#888'>ΤΑΣΗ (σκορ {trend['score']:+d})</div>"
            f"<div style='font-weight:800;color:{trend['color']}'>{trend['verdict']}</div></div>",
            unsafe_allow_html=True,
        )

        # --- ΑΠΛΗ ΕΞΗΓΗΣΗ + BUY/HOLD/SELL ΠΙΤΑ ---
        st.markdown("")  # μικρό κενό
        sum_left, sum_right = st.columns([1.4, 1])

        with sum_left:
            st.markdown("#### 💡 Με απλά λόγια")
            st.markdown(
                f"<div style='padding:14px 18px;border-radius:12px;background:{trend['color']}15;"
                f"border-left:4px solid {trend['color']};font-size:1.05rem;line-height:1.6'>"
                f"{simple_explanation(trend, last)}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Αυτή είναι μια αυτόματη σύνοψη των τεχνικών δεικτών (RSI, MACD, μέσοι όροι κ.λπ.) — "
                "όχι προσωπική γνώμη ή πρόβλεψη."
            )

        with sum_right:
            bhs = buy_hold_sell(trend["score"])
            pie = go.Figure(data=[go.Pie(
                labels=["Buy (Αγορά)", "Hold (Κράτημα)", "Sell (Πώληση)"],
                values=[bhs["Buy"], bhs["Hold"], bhs["Sell"]],
                hole=0.45,
                marker=dict(colors=["#1a9850", "#bdbdbd", "#d73027"]),
                textinfo="label+percent",
                textfont=dict(size=13),
                sort=False,
            )])
            # Ποια ετικέτα κυριαρχεί
            dominant = max(bhs, key=bhs.get)
            dom_color = {"Buy": "#1a9850", "Hold": "#999999", "Sell": "#d73027"}[dominant]
            pie.update_layout(
                height=300, margin=dict(t=40, b=10, l=10, r=10),
                showlegend=False, template="plotly_white",
                title=dict(text="Σήμα ανάλυσης", x=0.5, font=dict(size=15)),
                annotations=[dict(text=f"<b>{dominant}</b>", x=0.5, y=0.5,
                                  font=dict(size=18, color=dom_color), showarrow=False)],
            )
            st.plotly_chart(pie, use_container_width=True)

        st.divider()

        # --- Tabs ---
        tab_chart, tab_tech, tab_fund, tab_macro = st.tabs(
            ["📊 Γράφημα", "🔧 Τεχνικά σήματα", "🏢 Θεμελιώδη", "🌍 Μακρο & Συνεδριάσεις"]
        )

        # ===== TAB 1: Γράφημα =====
        with tab_chart:
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.17, 0.17, 0.16], vertical_spacing=0.03,
                subplot_titles=("Τιμή + Bollinger + SMA", "Volume", "RSI", "MACD"),
            )
            # Candles
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Τιμή", increasing_line_color="#1a9850", decreasing_line_color="#d73027",
            ), row=1, col=1)
            for col, color, w in [("SMA50", "#2166ac", 1.2), ("SMA200", "#b2182b", 1.2), ("MA20", "#f1a340", 1)]:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=w)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], name="BB up", line=dict(color="#888", width=0.7, dash="dot"), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], name="BB low", line=dict(color="#888", width=0.7, dash="dot"), fill="tonexty", fillcolor="rgba(120,120,120,0.08)", showlegend=False), row=1, col=1)
            # Volume
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#aaa"), row=2, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#762a83")), row=3, col=1)
            fig.add_hline(y=70, line=dict(color="#d73027", width=0.7, dash="dash"), row=3, col=1)
            fig.add_hline(y=30, line=dict(color="#1a9850", width=0.7, dash="dash"), row=3, col=1)
            # MACD
            colors = ["#1a9850" if v >= 0 else "#d73027" for v in df["MACD_hist"].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist", marker_color=colors), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2166ac", width=1)), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="#f1a340", width=1)), row=4, col=1)

            fig.update_layout(height=820, template="plotly_white", xaxis_rangeslider_visible=False,
                              legend=dict(orientation="h", y=1.04), margin=dict(t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # ===== TAB 2: Τεχνικά σήματα =====
        with tab_tech:
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("SMA50", f"{last['SMA50']:.2f}" if not np.isnan(last['SMA50']) else "—")
            cc2.metric("SMA200", f"{last['SMA200']:.2f}" if not np.isnan(last['SMA200']) else "—")
            cc3.metric("ATR (14)", f"{last['ATR']:.2f}" if not np.isnan(last['ATR']) else "—",
                       help="Μέτρο μεταβλητότητας — μέσο εύρος κίνησης")
            cc4.metric("MACD", f"{last['MACD']:.2f}" if not np.isnan(last['MACD']) else "—")

            st.subheader("Τι σημαίνει κάθε δείκτης για αυτή τη μετοχή")
            st.caption("Κάθε δείκτης εξηγείται με απλά λόγια, βάσει της σημερινής του τιμής.")

            for title, value, explanation in explain_indicators(last, df):
                st.markdown(
                    f"<div style='padding:12px 16px;margin-bottom:10px;border-radius:10px;"
                    f"background:#e2e5ea;border-left:3px solid #2166ac'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:700;font-size:1.02rem;color:#000000'>{title}</span>"
                    f"<span style='font-weight:700;color:#2166ac'>{value}</span></div>"
                    f"<div style='margin-top:6px;color:#333;line-height:1.55'>{explanation}</div></div>",
                    unsafe_allow_html=True,
                )

            st.info(
                f"**Συνολική εκτίμηση: {trend['verdict']}** (σκορ {trend['score']:+d}/100). "
                "Το σκορ συνδυάζει όλους τους παραπάνω δείκτες. "
                "Δεν είναι πρόβλεψη — είναι σύνοψη των τρεχόντων τεχνικών σημάτων."
            )

        # ===== TAB 3: Θεμελιώδη =====
        with tab_fund:
            if not info:
                st.warning("Δεν υπάρχουν θεμελιώδη στοιχεία (εμπόρευμα, κρυπτονόμισμα, ETF ή index — όχι εταιρεία). Η τεχνική ανάλυση όμως ισχύει κανονικά.")
            else:
                f1, f2, f3 = st.columns(3)
                f1.metric("Market Cap", fmt_num(info.get("marketCap")))
                f2.metric("P/E (trailing)", f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else "—")
                f3.metric("EPS", f"{info.get('trailingEps'):.2f}" if info.get("trailingEps") else "—")

                f4, f5, f6 = st.columns(3)
                f4.metric("Έσοδα (TTM)", fmt_num(info.get("totalRevenue")))
                f5.metric("Free Cash Flow", fmt_num(info.get("freeCashflow")))
                f6.metric("Operating CF", fmt_num(info.get("operatingCashflow")))

                f7, f8, f9 = st.columns(3)
                f7.metric("Μερισματική απόδ.", f"{info.get('dividendYield')*100:.2f}%" if info.get("dividendYield") else "—")
                f8.metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "—")
                f9.metric("Profit Margin", f"{info.get('profitMargins')*100:.1f}%" if info.get("profitMargins") else "—")

                target = info.get("targetMeanPrice")
                if target:
                    upside = (target - price) / price * 100
                    st.markdown(
                        f"🎯 **Μέση τιμή-στόχος αναλυτών:** {target:.2f} {currency} "
                        f"({upside:+.1f}% από τρέχουσα) · Σύσταση: **{info.get('recommendationKey', '—')}**"
                    )

                if cashflow is not None and not cashflow.empty:
                    with st.expander("📋 Πίνακας Cash Flow (ετήσιος)"):
                        st.dataframe((cashflow / 1e6).round(0).rename_axis("σε εκ.").style.format("{:,.0f}"),
                                     use_container_width=True)

                summary = info.get("longBusinessSummary")
                if summary:
                    with st.expander("ℹ️ Περιγραφή εταιρείας"):
                        st.write(summary)

        # ===== TAB 4: Μακρο & Συνεδριάσεις =====
        with tab_macro:
            st.subheader("Μακροοικονομικό υπόβαθρο")
            macro = load_macro()
            if macro:
                mcols = st.columns(len(macro))
                for col, (k, (val, ch)) in zip(mcols, macro.items()):
                    col.metric(k, f"{val:.2f}", f"{ch:+.1f}% / μήνα")
            st.caption("10Y yield ↑ → πίεση σε μετοχές · VIX ↑ → φόβος/μεταβλητότητα · DXY ↑ → ισχυρό δολάριο")

            st.divider()
            st.subheader("📅 Επόμενες συνεδριάσεις κεντρικών τραπεζών")
            nfomc = next_meeting(FOMC_DATES)
            necb = next_meeting(ECB_DATES)
            mc1, mc2 = st.columns(2)
            if nfomc:
                days = (nfomc - dt.date.today()).days
                mc1.metric("🇺🇸 Επόμενη FOMC (Fed)", nfomc.strftime("%d %b %Y"), f"σε {days} ημέρες")
            if necb:
                days = (necb - dt.date.today()).days
                mc2.metric("🇪🇺 Επόμενη ΕΚΤ (ECB)", necb.strftime("%d %b %Y"), f"σε {days} ημέρες")
            st.info(
                "Οι συνεδριάσεις κεντρικών τραπεζών συχνά αυξάνουν τη μεταβλητότητα. "
                "Αν πλησιάζει συνεδρίαση, οι αποφάσεις επιτοκίων μπορεί να επηρεάσουν έντονα την αγορά "
                "ανεξάρτητα από τα τεχνικά σήματα της μετοχής."
            )

    except Exception as e:
        st.error(f"Σφάλμα κατά την ανάλυση: {e}")
        st.caption("Δοκίμασε άλλο σύμβολο ή ξαναπροσπάθησε σε λίγο (πιθανό rate-limit του Yahoo).")
