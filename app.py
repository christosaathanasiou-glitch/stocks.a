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


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    """Stochastic Oscillator (%K, %D) — 0-100, δείχνει υπεραγορά/υπερπώληση."""
    low_k = df["Low"].rolling(k).min()
    high_k = df["High"].rolling(k).max()
    pct_k = 100 * (df["Close"] - low_k) / (high_k - low_k)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def adx(df: pd.DataFrame, window: int = 14):
    """Average Directional Index — δύναμη τάσης (0-100, >25 = ισχυρή τάση)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx_, plus_di, minus_di


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — αθροιστικός όγκος ανάλογα με άνοδο/πτώση."""
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).fillna(0).cumsum()


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index — απόκλιση από τον μέσο (±100 όρια)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md)


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R — υπεραγορά/υπερπώληση (-100 έως 0)."""
    high = df["High"].rolling(window).max()
    low = df["Low"].rolling(window).min()
    return -100 * (high - df["Close"]) / (high - low)


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change — ποσοστιαία μεταβολή τιμής (momentum)."""
    return (series - series.shift(window)) / series.shift(window) * 100


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index — «RSI με όγκο» (0-100)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    mf = tp * df["Volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(window).sum()
    neg_sum = neg.rolling(window).sum()
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow — πίεση αγοράς/πώλησης μέσω όγκου (-1 έως +1)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfv = mfm * df["Volume"]
    return mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει όλους τους δείκτες σε ένα DataFrame OHLCV."""
    out = df.copy()
    close = out["Close"]
    # Μέσοι όροι & βασικοί
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["MA20"] = sma(close, 20)
    out["EMA20"] = ema(close, 20)
    out["RSI"] = rsi(close, 14)
    out["ATR"] = atr(out, 14)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(close)
    out["BB_mid"], out["BB_up"], out["BB_low"] = bollinger(close)
    # Επιπλέον δείκτες (για πιο ακριβή ανάλυση)
    out["STOCH_K"], out["STOCH_D"] = stochastic(out)
    out["ADX"], out["DI_plus"], out["DI_minus"] = adx(out)
    out["OBV"] = obv(out)
    out["OBV_ema"] = ema(out["OBV"], 20)
    out["CCI"] = cci(out)
    out["WILLR"] = williams_r(out)
    out["ROC"] = roc(close, 12)
    out["MFI"] = mfi(out)
    out["CMF"] = cmf(out)
    out["VOL_avg"] = sma(out["Volume"], 20)
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

    # 6. ADX — δύναμη τάσης (ενισχύει την κατεύθυνση των DI)
    if not np.isnan(last.get("ADX", np.nan)):
        if last["ADX"] > 25:
            if last["DI_plus"] > last["DI_minus"]:
                score += 12; signals.append((f"ADX {last['ADX']:.0f}", "bullish", "Ισχυρή ανοδική τάση"))
            else:
                score -= 12; signals.append((f"ADX {last['ADX']:.0f}", "bearish", "Ισχυρή καθοδική τάση"))

    # 7. Stochastic
    if not np.isnan(last.get("STOCH_K", np.nan)):
        k = last["STOCH_K"]
        if k > 80:
            score -= 6; signals.append((f"Stochastic {k:.0f}", "bearish", "Υπεραγορασμένη ζώνη"))
        elif k < 20:
            score += 6; signals.append((f"Stochastic {k:.0f}", "bullish", "Υπερπουλημένη ζώνη"))

    # 8. MFI (money flow)
    if not np.isnan(last.get("MFI", np.nan)):
        m = last["MFI"]
        if m > 80:
            score -= 5; signals.append((f"MFI {m:.0f}", "bearish", "Υπερβολική εισροή χρήματος"))
        elif m < 20:
            score += 5; signals.append((f"MFI {m:.0f}", "bullish", "Υπερβολική εκροή — πιθανή ανάκαμψη"))

    # 9. CCI
    if not np.isnan(last.get("CCI", np.nan)):
        c = last["CCI"]
        if c > 100:
            score += 4; signals.append((f"CCI {c:.0f}", "bullish", "Ισχυρή ανοδική ορμή"))
        elif c < -100:
            score -= 4; signals.append((f"CCI {c:.0f}", "bearish", "Ισχυρή καθοδική ορμή"))

    # 10. Williams %R
    if not np.isnan(last.get("WILLR", np.nan)):
        w = last["WILLR"]
        if w > -20:
            score -= 4; signals.append((f"Williams %R {w:.0f}", "bearish", "Υπεραγορασμένη"))
        elif w < -80:
            score += 4; signals.append((f"Williams %R {w:.0f}", "bullish", "Υπερπουλημένη"))

    # 11. ROC (momentum)
    if not np.isnan(last.get("ROC", np.nan)):
        rc = last["ROC"]
        if rc > 0:
            score += 4; signals.append((f"ROC {rc:+.1f}%", "bullish", "Θετική ορμή τιμής"))
        else:
            score -= 4; signals.append((f"ROC {rc:+.1f}%", "bearish", "Αρνητική ορμή τιμής"))

    # 12. OBV vs EMA (τάση όγκου)
    if not np.isnan(last.get("OBV", np.nan)) and not np.isnan(last.get("OBV_ema", np.nan)):
        if last["OBV"] > last["OBV_ema"]:
            score += 5; signals.append(("OBV ανοδικό", "bullish", "Ο όγκος στηρίζει την άνοδο"))
        else:
            score -= 5; signals.append(("OBV καθοδικό", "bearish", "Ο όγκος στηρίζει την πτώση"))

    # 13. CMF (Chaikin money flow)
    if not np.isnan(last.get("CMF", np.nan)):
        cm = last["CMF"]
        if cm > 0.05:
            score += 4; signals.append(("CMF θετικό", "bullish", "Πίεση αγοράς"))
        elif cm < -0.05:
            score -= 4; signals.append(("CMF αρνητικό", "bearish", "Πίεση πώλησης"))

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


def _all_indicator_explanations(last, close) -> list:
    """Χτίζει εξήγηση για ΚΑΘΕ δείκτη μαζί με βαθμό σημαντικότητας (0-10).
    Επιστρέφει λίστα από dicts: {title, value, text, importance, kind}.
    Όσο πιο ακραίο/καθαρό το σήμα ενός δείκτη, τόσο μεγαλύτερη η σημαντικότητα,
    ώστε να επιλέγονται αυτόματα οι πιο κρίσιμοι για τη συγκεκριμένη αγορά."""
    out = []

    def add(title, value, text, importance, kind="neutral"):
        out.append({"title": title, "value": value, "text": text,
                    "importance": importance, "kind": kind})

    # --- RSI ---
    r = last.get("RSI", np.nan)
    if not np.isnan(r):
        if r > 70:
            add("RSI (δείκτης δύναμης)", f"{r:.0f}",
                f"Το RSI είναι {r:.0f}, πάνω από 70 («υπεραγορασμένη»): ανέβηκε πολύ γρήγορα και "
                "ίσως χρειαστεί ξεκούραση ή μικρή πτώση. Σαν δρομέα που λαχανιάζει.", 9, "bearish")
        elif r < 30:
            add("RSI (δείκτης δύναμης)", f"{r:.0f}",
                f"Το RSI είναι {r:.0f}, κάτω από 30 («υπερπουλημένη»): έπεσε πολύ και ίσως είναι "
                "έτοιμη να ανακάμψει. Σαν ελατήριο που πιέστηκε και θέλει να πεταχτεί.", 9, "bullish")
        elif r >= 50:
            add("RSI (δείκτης δύναμης)", f"{r:.0f}",
                f"Το RSI είναι {r:.0f}, λίγο πάνω από τη μέση (50): οι αγοραστές έχουν ελαφρύ "
                "προβάδισμα — υγιές, χωρίς υπερβολές.", 4, "bullish")
        else:
            add("RSI (δείκτης δύναμης)", f"{r:.0f}",
                f"Το RSI είναι {r:.0f}, λίγο κάτω από τη μέση (50): οι πωλητές έχουν ελαφρύ "
                "προβάδισμα, αλλά τίποτα ακραίο.", 4, "bearish")

    # --- SMA50 vs SMA200 ---
    s50, s200 = last.get("SMA50", np.nan), last.get("SMA200", np.nan)
    if not (np.isnan(s50) or np.isnan(s200)):
        gap = abs(s50 - s200) / s200 * 100 if s200 else 0
        imp = 8 if gap > 3 else 6
        if s50 > s200:
            add("Μέσοι όροι (SMA50 vs SMA200)", "Ανοδική",
                f"Ο μέσος όρος 50 ημερών ({s50:.2f}) είναι πάνω από τον 200 ημερών ({s200:.2f}) — "
                "«golden cross», δηλαδή ανοδική τάση: η πρόσφατη πορεία είναι καλύτερη από τη μακροχρόνια.",
                imp, "bullish")
        else:
            add("Μέσοι όροι (SMA50 vs SMA200)", "Καθοδική",
                f"Ο μέσος όρος 50 ημερών ({s50:.2f}) είναι κάτω από τον 200 ημερών ({s200:.2f}) — "
                "«death cross», δηλαδή καθοδική τάση: η πρόσφατη πορεία είναι χειρότερη από τη μακροχρόνια.",
                imp, "bearish")

    # --- Τιμή vs SMA200 ---
    if not np.isnan(s200):
        if close > s200:
            add("Τιμή σε σχέση με SMA200", "Πάνω",
                f"Η τιμή ({close:.2f}) είναι πάνω από τον μέσο όρο 200 ημερών ({s200:.2f}): "
                "μακροπρόθεσμα ανοδικό «κλίμα».", 7, "bullish")
        else:
            add("Τιμή σε σχέση με SMA200", "Κάτω",
                f"Η τιμή ({close:.2f}) είναι κάτω από τον μέσο όρο 200 ημερών ({s200:.2f}): "
                "μακροπρόθεσμα καθοδικό «κλίμα».", 7, "bearish")

    # --- MACD ---
    m, sig = last.get("MACD", np.nan), last.get("MACD_signal", np.nan)
    if not (np.isnan(m) or np.isnan(sig)):
        if m > sig:
            add("MACD (φόρα/μομέντουμ)", "Ανοδικό",
                "Η γραμμή MACD είναι πάνω από τη «signal»: το μομέντουμ (η «φόρα») στρέφεται προς τα πάνω.",
                7, "bullish")
        else:
            add("MACD (φόρα/μομέντουμ)", "Καθοδικό",
                "Η γραμμή MACD είναι κάτω από τη «signal»: η «φόρα» στρέφεται προς τα κάτω.",
                7, "bearish")

    # --- Bollinger Bands ---
    bb_up, bb_low, bb_mid = last.get("BB_up", np.nan), last.get("BB_low", np.nan), last.get("BB_mid", np.nan)
    if not (np.isnan(bb_up) or np.isnan(bb_low)):
        if close > bb_up:
            add("Bollinger Bands (εύρος τιμής)", "Πάνω μπάντα",
                "Η τιμή ξεπέρασε την πάνω μπάντα: ανέβηκε απότομα και ίσως «επιστρέψει» λίγο προς τα κάτω.",
                7, "bearish")
        elif close < bb_low:
            add("Bollinger Bands (εύρος τιμής)", "Κάτω μπάντα",
                "Η τιμή έπεσε κάτω από την κάτω μπάντα: έπεσε απότομα και ίσως «αναπηδήσει» προς τα πάνω.",
                7, "bullish")
        else:
            add("Bollinger Bands (εύρος τιμής)", "Εντός εύρους",
                f"Η τιμή κινείται ανάμεσα στις μπάντες (γύρω από το μέσο {bb_mid:.2f}): κανονικό εύρος, "
                "καμία ακραία κίνηση τώρα.", 3, "neutral")

    # --- ADX (δύναμη τάσης) ---
    ax = last.get("ADX", np.nan)
    if not np.isnan(ax):
        dip, dim = last.get("DI_plus", np.nan), last.get("DI_minus", np.nan)
        if ax > 25:
            direction = "ανοδική" if dip > dim else "καθοδική"
            add("ADX (δύναμη τάσης)", f"{ax:.0f}",
                f"Το ADX είναι {ax:.0f} (>25): υπάρχει ισχυρή {direction} τάση — η κίνηση έχει «καύσιμο» "
                "και δεν είναι απλώς θόρυβος.", 8, "bullish" if dip > dim else "bearish")
        else:
            add("ADX (δύναμη τάσης)", f"{ax:.0f}",
                f"Το ADX είναι {ax:.0f} (<25): αδύναμη ή ανύπαρκτη τάση — η τιμή μάλλον κινείται "
                "πλάγια, χωρίς ξεκάθαρη κατεύθυνση.", 5, "neutral")

    # --- Stochastic ---
    k = last.get("STOCH_K", np.nan)
    if not np.isnan(k):
        if k > 80:
            add("Stochastic (ταχύτητα)", f"{k:.0f}",
                f"Το Stochastic είναι {k:.0f} (>80): υπεραγορασμένη ζώνη — πιθανή κόπωση της ανόδου.",
                6, "bearish")
        elif k < 20:
            add("Stochastic (ταχύτητα)", f"{k:.0f}",
                f"Το Stochastic είναι {k:.0f} (<20): υπερπουλημένη ζώνη — πιθανή ανάκαμψη.",
                6, "bullish")
        else:
            add("Stochastic (ταχύτητα)", f"{k:.0f}",
                f"Το Stochastic είναι {k:.0f}: ουδέτερη ζώνη, χωρίς ακραίο σήμα.", 2, "neutral")

    # --- MFI (money flow) ---
    mf = last.get("MFI", np.nan)
    if not np.isnan(mf):
        if mf > 80:
            add("MFI (ροή χρήματος)", f"{mf:.0f}",
                f"Το MFI είναι {mf:.0f} (>80): υπερβολική εισροή χρήματος — προσοχή σε πιθανή διόρθωση.",
                6, "bearish")
        elif mf < 20:
            add("MFI (ροή χρήματος)", f"{mf:.0f}",
                f"Το MFI είναι {mf:.0f} (<20): υπερβολική εκροή χρήματος — πιθανή ανάκαμψη.",
                6, "bullish")
        else:
            add("MFI (ροή χρήματος)", f"{mf:.0f}",
                f"Το MFI είναι {mf:.0f}: ισορροπημένη ροή χρήματος, χωρίς ακραίο σήμα.", 2, "neutral")

    # --- CCI ---
    c = last.get("CCI", np.nan)
    if not np.isnan(c):
        if c > 100:
            add("CCI (ορμή)", f"{c:.0f}",
                f"Το CCI είναι {c:.0f} (>100): ισχυρή ανοδική ορμή, αλλά και πιθανή υπεραγορά.", 5, "bullish")
        elif c < -100:
            add("CCI (ορμή)", f"{c:.0f}",
                f"Το CCI είναι {c:.0f} (<-100): ισχυρή καθοδική ορμή, αλλά και πιθανή υπερπώληση.", 5, "bearish")
        else:
            add("CCI (ορμή)", f"{c:.0f}",
                f"Το CCI είναι {c:.0f}: εντός κανονικού εύρους (±100).", 2, "neutral")

    # --- Williams %R ---
    w = last.get("WILLR", np.nan)
    if not np.isnan(w):
        if w > -20:
            add("Williams %R", f"{w:.0f}",
                f"Το Williams %R είναι {w:.0f} (>-20): υπεραγορασμένη — πιθανή κόπωση.", 5, "bearish")
        elif w < -80:
            add("Williams %R", f"{w:.0f}",
                f"Το Williams %R είναι {w:.0f} (<-80): υπερπουλημένη — πιθανή ανάκαμψη.", 5, "bullish")

    # --- ROC (momentum) ---
    rc = last.get("ROC", np.nan)
    if not np.isnan(rc):
        strong = abs(rc) > 8
        add("ROC (ορμή τιμής)", f"{rc:+.1f}%",
            f"Η τιμή {'ανέβηκε' if rc > 0 else 'έπεσε'} {abs(rc):.1f}% σε ~2 εβδομάδες — "
            f"{'έντονη' if strong else 'ήπια'} {'θετική' if rc > 0 else 'αρνητική'} ορμή.",
            6 if strong else 3, "bullish" if rc > 0 else "bearish")

    # --- OBV (τάση όγκου) ---
    ob, obe = last.get("OBV", np.nan), last.get("OBV_ema", np.nan)
    if not (np.isnan(ob) or np.isnan(obe)):
        if ob > obe:
            add("OBV (όγκος)", "Ανοδικό",
                "Ο συσσωρευμένος όγκος (OBV) ανεβαίνει: ο όγκος συναλλαγών στηρίζει την άνοδο — καλό σημάδι.",
                5, "bullish")
        else:
            add("OBV (όγκος)", "Καθοδικό",
                "Ο συσσωρευμένος όγκος (OBV) πέφτει: ο όγκος στηρίζει την πτώση.", 5, "bearish")

    # --- CMF ---
    cm = last.get("CMF", np.nan)
    if not np.isnan(cm):
        if cm > 0.05:
            add("CMF (πίεση αγοράς)", f"{cm:+.2f}",
                f"Το CMF είναι {cm:+.2f} (θετικό): επικρατεί πίεση αγοράς — τα «έξυπνα» χρήματα μπαίνουν.",
                4, "bullish")
        elif cm < -0.05:
            add("CMF (πίεση αγοράς)", f"{cm:+.2f}",
                f"Το CMF είναι {cm:+.2f} (αρνητικό): επικρατεί πίεση πώλησης.", 4, "bearish")

    # --- ATR (μεταβλητότητα) ---
    a = last.get("ATR", np.nan)
    if not np.isnan(a):
        pct = a / close * 100 if close else 0
        if pct > 4:
            level, extra, imp = "υψηλή", "Κινείται έντονα — μεγαλύτερο ρίσκο αλλά και ευκαιρίες.", 5
        elif pct > 2:
            level, extra, imp = "μέτρια", "Φυσιολογικές καθημερινές διακυμάνσεις.", 3
        else:
            level, extra, imp = "χαμηλή", "Κινείται ήρεμα, με μικρές καθημερινές αλλαγές.", 3
        add("ATR (μεταβλητότητα)", f"{a:.2f}",
            f"Το ATR είναι {a:.2f} (~{pct:.1f}% της τιμής): {level} μεταβλητότητα — πόσο «νευρικά» "
            f"κινείται η τιμή. {extra}", imp, "neutral")

    return out


def explain_indicators(last, df, top_n: int = 5) -> list:
    """Επιστρέφει μόνο τους top_n πιο σημαντικούς δείκτες για τη συγκεκριμένη αγορά,
    ως λίστα από (τίτλος, τιμή, εξήγηση). Η ανάλυση γίνεται με ΟΛΟΥΣ τους δείκτες,
    αλλά εμφανίζονται μόνο οι πιο κρίσιμοι αυτή τη στιγμή."""
    allx = _all_indicator_explanations(last, last["Close"])
    # Ταξινόμηση κατά σημαντικότητα (φθίνουσα), κρατάμε τους top_n
    allx.sort(key=lambda d: d["importance"], reverse=True)
    top = allx[:top_n]
    return [(d["title"], d["value"], d["text"]) for d in top]


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
    /* Κρύβει το πάνω-δεξιά μενού (☰), την κεφαλίδα και το footer του Streamlit */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    div[data-testid="stToolbar"] { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }
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
            chart_type = st.radio(
                "Τύπος γραφήματος",
                ["Κεριά (candlesticks)", "Γραμμή"],
                horizontal=True,
                index=0,
            )

            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.17, 0.17, 0.16], vertical_spacing=0.03,
                subplot_titles=("Τιμή + Bollinger + SMA", "Volume", "RSI", "MACD"),
            )
            # Τιμή: κεριά ή απλή γραμμή, ανάλογα με την επιλογή
            if chart_type.startswith("Κεριά"):
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="Τιμή", increasing_line_color="#1a9850", decreasing_line_color="#d73027",
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["Close"], name="Τιμή (κλείσιμο)",
                    line=dict(color="#111111", width=1.6),
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

            st.subheader("Οι πιο σημαντικοί δείκτες αυτή τη στιγμή")
            st.caption("Η ανάλυση γίνεται με 14+ δείκτες· εδώ εμφανίζονται οι 5 πιο κρίσιμοι για τη συγκεκριμένη επιλογή, με απλά λόγια.")

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
