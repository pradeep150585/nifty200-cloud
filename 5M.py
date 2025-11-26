"""
Daily_Scanner_fixed_investing_style.py

Updated: indicator logic carefully aligned with Investing.com conventions:
 - RSI(14) thresholds 30/70
 - Stochastic (9,3,3) thresholds 20/80
 - StochRSI(14) thresholds 20/80
 - MACD (12,26,9) cross detection
 - ADX/DMI (Wilder) with ADX>25 rule
 - Williams %R (14) -80/-20
 - CCI (14) ±100
 - Ultimate Oscillator (7,14,28) weights 4/2/1 threshold 30/70
 - ROC(12) zero-cross detection and sign fallback
 - EMA(13) used for Elder-Ray
 - Rolling min_periods set to indicator period where applicable
"""

import warnings
warnings.simplefilter("ignore")

import io
import math
import time
import requests
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
YFINANCE_PERIOD = "14d"
YFINANCE_INTERVAL = "5m"
OUTPUT_FILENAME = "Nifty200_Weighted_Balanced_5m_fixed.xlsx"
BATCH_SIZE = 20
MOM_PERIOD = 10
MIN_ROWS_REQUIRED = 80
MAX_WORKERS = 6
SLEEP_BETWEEN_BATCHES = 0.5

# -----------------------
# UTILITIES
# -----------------------
def fetch_nifty200_symbols() -> List[str]:
    r = requests.get(NIFTY200_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    sym_col = [c for c in df.columns if "symbol" in c.lower()][0]
    symbols = df[sym_col].astype(str).str.strip().tolist()
    return [s.upper() + ".NS" for s in symbols]

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def safe_div(a, b):
     try:
          if hasattr(b, "replace"):
               b_nonzero = b.replace(0, np.nan)
               return a / b_nonzero
          if b == 0:
               return np.nan
          return a / b
     except Exception:
          return np.nan

def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing using alpha = 1/period (commonly used for RSI/ADX)."""
    return series.ewm(alpha=1.0/period, adjust=False).mean()

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))

def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

# -----------------------
# INDICATORS (Investing.com-style)
# -----------------------
def compute_indicators_vectorized(df: pd.DataFrame, mom_period=MOM_PERIOD) -> pd.DataFrame:
    """
    Compute indicators and produce *_Rating columns using Investing.com standard thresholds.
    """
    # required columns
    if not {"Close", "High", "Low"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Close', 'High', 'Low' columns.")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # -----------------------
    # RSI(14) (Wilder)
    # -----------------------
    period_rsi = 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period_rsi)
    avg_loss = wilder_smooth(loss, period_rsi)
    RS = safe_div(avg_gain, avg_loss)
    df["RSI"] = 100 - (100 / (1 + RS))
    # strict Investing.com thresholds
    df["RSI_Rating"] = df["RSI"].apply(lambda v: "Buy" if (not pd.isna(v) and v < 30) else ("Sell" if (not pd.isna(v) and v > 70) else "Neutral"))

    # -----------------------
    # Stochastic Oscillator (9,3,3)
    # %K raw, %K smoothed (SMA3), %D (SMA3 of %K_smooth)
    # -----------------------
    period_stoch = 9
    minp_stoch = period_stoch
    low9 = low.rolling(period_stoch, min_periods=minp_stoch).min()
    high9 = high.rolling(period_stoch, min_periods=minp_stoch).max()
    k_raw = safe_div((close - low9), (high9 - low9)) * 100
    # smooth %K by 3-period SMA (not ewm)
    k_smooth = k_raw.rolling(3, min_periods=3).mean()
    d_smooth = k_smooth.rolling(3, min_periods=3).mean()
    df["STOCH_%K"] = k_raw
    df["STOCH_%K_smooth"] = k_smooth
    df["STOCH_%D"] = d_smooth
    df["STO_Rating"] = df["STOCH_%K_smooth"].apply(lambda v: "Buy" if (not pd.isna(v) and v < 20) else ("Sell" if (not pd.isna(v) and v > 80) else "Neutral"))

    # -----------------------
    # StochRSI (14)
    # -----------------------
    # Use RSI(14) min/max over last 14 periods
    stochrsi_period = 14
    rsi_min = df["RSI"].rolling(stochrsi_period, min_periods=stochrsi_period).min()
    rsi_max = df["RSI"].rolling(stochrsi_period, min_periods=stochrsi_period).max()
    stochrsi = safe_div(df["RSI"] - rsi_min, (rsi_max - rsi_min)) * 100
    df["StochRSI"] = stochrsi
    df["StochRSI_Rating"] = df["StochRSI"].apply(lambda v: "Buy" if (not pd.isna(v) and v < 20) else ("Sell" if (not pd.isna(v) and v > 80) else "Neutral"))

    # -----------------------
    # MACD (12,26,9) crossover
    # -----------------------
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd - macd_signal
    df["MACD_Rating"] = "Neutral"
    df.loc[cross_up(df["MACD"], df["MACD_Signal"]), "MACD_Rating"] = "Buy"
    df.loc[cross_down(df["MACD"], df["MACD_Signal"]), "MACD_Rating"] = "Sell"

    # -----------------------
    # True Range & ATR (Wilder 14)
    # -----------------------
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["TR"] = TR
    df["ATR"] = wilder_smooth(TR, 14)

    # -----------------------
    # ADX / DMI (Wilder 14)
    # -----------------------
    up_move = high.diff()
    down_move = -low.diff()
    DMp = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    DMm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    DMp_w = wilder_smooth(DMp, 14)
    DMm_w = wilder_smooth(DMm, 14)
    ATR_w = df["ATR"]
    DIp = 100 * safe_div(DMp_w, ATR_w)
    DIm = 100 * safe_div(DMm_w, ATR_w)
    DX = 100 * safe_div((DIp - DIm).abs(), (DIp + DIm))
    ADX = wilder_smooth(DX, 14)
    df["DIp"] = DIp
    df["DIm"] = DIm
    df["ADX"] = ADX

    def adx_rating_fn(adx_val, dip_val, dim_val):
        if pd.isna(adx_val) or pd.isna(dip_val) or pd.isna(dim_val):
            return "Neutral"
        # Investing.com style: ADX > 25 means a trend; choose DI+ vs DI-
        if adx_val > 25:
            return "Buy" if dip_val > dim_val else "Sell"
        return "Neutral"

    df["ADX_Rating"] = [adx_rating_fn(a, b, c) for a, b, c in zip(df["ADX"], df["DIp"], df["DIm"])]

    # -----------------------
    # Williams %R (14)
    # -----------------------
    period_wr = 14
    hh14 = high.rolling(period_wr, min_periods=period_wr).max()
    ll14 = low.rolling(period_wr, min_periods=period_wr).min()
    # Williams %R standard formula
    df["WR"] = -100 * safe_div(hh14 - close, (hh14 - ll14))
    df["WR_Rating"] = df["WR"].apply(lambda v: "Buy" if (not pd.isna(v) and v < -80) else ("Sell" if (not pd.isna(v) and v > -20) else "Neutral"))

    # -----------------------
    # CCI (14)
    # -----------------------
    period_cci = 14
    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(period_cci, min_periods=period_cci).mean()
    mad = (tp - tp_sma).abs().rolling(period_cci, min_periods=period_cci).mean()
    df["CCI"] = safe_div((tp - tp_sma), (0.015 * mad))
    df["CCI_Rating"] = df["CCI"].apply(lambda v: "Buy" if (not pd.isna(v) and v < -100) else ("Sell" if (not pd.isna(v) and v > 100) else "Neutral"))

    # -----------------------
    # High/Low breakout (14)
    # -----------------------
    df["HL_High14"] = hh14
    df["HL_Low14"] = ll14
    def highlow_rating(row):
        c = row["Close"]
        hh = row["HL_High14"]
        ll = row["HL_Low14"]
        if pd.isna(c) or pd.isna(hh) or pd.isna(ll):
            return "Neutral"
        if c > hh:
            return "Buy"
        if c < ll:
            return "Sell"
        return "Neutral"
    df["HighLow_Rating"] = df.apply(highlow_rating, axis=1)

    # -----------------------
    # Ultimate Oscillator (7,14,28)
    # -----------------------
    prev_close = close.shift(1)
    bp = close - np.minimum(low, prev_close)
    tr_uo = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    sum7 = bp.rolling(7, min_periods=7).sum(); tr7 = tr_uo.rolling(7, min_periods=7).sum()
    sum14 = bp.rolling(14, min_periods=14).sum(); tr14 = tr_uo.rolling(14, min_periods=14).sum()
    sum28 = bp.rolling(28, min_periods=28).sum(); tr28 = tr_uo.rolling(28, min_periods=28).sum()
    tr7 = tr_uo.rolling(7, min_periods=7).sum()
    tr14 = tr_uo.rolling(14, min_periods=14).sum()
    tr28 = tr_uo.rolling(28, min_periods=28).sum()
    avg7 = safe_div(sum7, tr7)
    avg14 = safe_div(sum14, tr14)
    avg28 = safe_div(sum28, tr28)
    df["UO"] = safe_div((4 * avg7 + 2 * avg14 + avg28), 7) * 100
    df["UO_Rating"] = "Neutral"
    df.loc[cross_up(df["UO"], pd.Series(50, index=df.index)), "UO_Rating"] = "Buy"
    df.loc[cross_down(df["UO"], pd.Series(50, index=df.index)), "UO_Rating"] = "Sell"
    # Fallback to band if no crossover yet
    mask_neutral = df["UO_Rating"] == "Neutral"
    df.loc[mask_neutral, "UO_Rating"] = df.loc[mask_neutral, "UO"].apply(lambda v: "Buy" if (not pd.isna(v) and v > 50) else ("Sell" if (not pd.isna(v) and v < 50) else "Neutral"))

    # -----------------------
    # ROC (12)
    # -----------------------
    period_roc = 12
    df["ROC"] = safe_div((close - close.shift(period_roc)), close.shift(period_roc)) * 100
    df["ROC_Rating"] = "Neutral"
    df.loc[cross_up(df["ROC"], pd.Series(0, index=df.index)), "ROC_Rating"] = "Buy"
    df.loc[cross_down(df["ROC"], pd.Series(0, index=df.index)), "ROC_Rating"] = "Sell"
    # fallback to sign if no cross
    mask_neutral = df["ROC_Rating"] == "Neutral"
    df.loc[mask_neutral, "ROC_Rating"] = df.loc[mask_neutral, "ROC"].apply(lambda v: "Buy" if (not pd.isna(v) and v > 0) else ("Sell" if (not pd.isna(v) and v < 0) else "Neutral"))

    # -----------------------
    # Elder-Ray (Bull/Bear) using EMA(13)
    # -----------------------
    ema13 = close.ewm(span=13, adjust=False).mean()
    df["BullPower"] = high - ema13
    df["BearPower"] = low - ema13
    df["BBP_Rating"] = df["BullPower"].apply(lambda v: "Buy" if (not pd.isna(v) and v > 0) else ("Sell" if (not pd.isna(v) and v < 0) else "Neutral"))

    # -----------------------
    # SMA/EMA (5/10/20/50/100)
    # price > MA -> Buy else Sell
    # -----------------------
    mas = [5, 10, 20, 50, 100]
    for m in mas:
        df[f"SMA{m}"] = close.rolling(m, min_periods=m).mean()
        df[f"EMA{m}"] = close.ewm(span=m, adjust=False).mean()
        df[f"SMA{m}_Rating"] = np.where((~df["Close"].isna()) & (~df[f"SMA{m}"].isna()),
                                         np.where(df["Close"] > df[f"SMA{m}"], "Buy", "Sell"),
                                         "Neutral")
        df[f"EMA{m}_Rating"] = np.where((~df["Close"].isna()) & (~df[f"EMA{m}"].isna()),
                                         np.where(df["Close"] > df[f"EMA{m}"], "Buy", "Sell"),
                                         "Neutral")

    # -----------------------
    # Momentum (MOM)
    # -----------------------
    df[f"MOM_{mom_period}"] = close.diff(mom_period)
    df[f"MOM_{mom_period}_Rating"] = df[f"MOM_{mom_period}"].apply(lambda v: "Buy" if (not pd.isna(v) and v > 0) else ("Sell" if (not pd.isna(v) and v < 0) else "Neutral"))

    # -----------------------
    # Collect ratings and compute scores
    # -----------------------
    rating_cols = [c for c in df.columns if c.endswith("_Rating")]
    df_ratings = df[rating_cols].replace({"Buy": 1, "Sell": -1, "Neutral": 0})
    df["Net_Score"] = df_ratings.sum(axis=1).astype(int)

    # -----------------------
    # Balanced Weight Model (your existing weights retained)
    # -----------------------
    WEIGHTS = {}
    osc_cols = [
        "RSI_Rating", "STO_Rating", "StochRSI_Rating", "CCI_Rating",
        "WR_Rating", "ROC_Rating", "UO_Rating", f"MOM_{mom_period}_Rating"
    ]
    for c in osc_cols:
        WEIGHTS[c] = 1

    ma_cols = []
    for m in mas:
        ma_cols.append(f"SMA{m}_Rating")
        ma_cols.append(f"EMA{m}_Rating")
    WEIGHTS.update({c: 2 for c in ma_cols})
    WEIGHTS["HighLow_Rating"] = 2
    WEIGHTS["BBP_Rating"] = 2
    WEIGHTS["MACD_Rating"] = 3
    WEIGHTS["ADX_Rating"] = 3

    weighted_df = df_ratings.copy()
    for col in weighted_df.columns:
        weight = WEIGHTS.get(col, 1)
        weighted_df[col] = weighted_df[col] * weight

    df["Weighted_Net_Score"] = weighted_df.sum(axis=1).astype(int)

    max_possible = sum(abs(WEIGHTS.get(col, 1)) for col in rating_cols)

    def classify_summary(score):
        if max_possible == 0:
            return "Neutral"
        ratio = score / max_possible
        if ratio >= 0.6:
            return "Strong Buy"
        if ratio >= 0.2:
            return "Buy"
        if -0.2 < ratio < 0.2:
            return "Neutral"
        if ratio <= -0.2 and ratio > -0.6:
            return "Sell"
        return "Strong Sell"

    df["Summary"] = df["Weighted_Net_Score"].apply(classify_summary)
    return df

# -----------------------
# PROCESSING helpers
# -----------------------
def process_symbol_from_df(symbol: str, df_tk: pd.DataFrame):
    try:
        if df_tk.empty or len(df_tk) < MIN_ROWS_REQUIRED:
            return None
        df_tk = df_tk.sort_index()
        df_ind = compute_indicators_vectorized(df_tk.copy())
        last = df_ind.iloc[-1]
        net_score = int(last["Net_Score"])
        weighted_score = int(last["Weighted_Net_Score"])
        summary = last["Summary"]
        cmp_price = float(last["Close"])
        prev_close = float(df_ind["Close"].iloc[-2]) if len(df_ind) >= 2 else np.nan
        change_pct = round((cmp_price - prev_close) / prev_close * 100, 2) if (prev_close and not math.isnan(prev_close) and prev_close != 0) else None
        return {"Symbol": symbol.replace(".NS", ""), "CMP": round(cmp_price,2), "Change%": change_pct, "Net_Score": net_score, "Weighted_Net_Score": weighted_score, "Summary": summary}
    except Exception:
        return None

def process_symbol_download(symbol: str):
    try:
        df_tk = yf.download(symbol, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, progress=False)
        if df_tk.empty or len(df_tk) < MIN_ROWS_REQUIRED:
            return None

        df_tk = df_tk.sort_index()
        return process_symbol_from_df(symbol, df_tk)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

# -----------------------
# RUN SCANNER
# -----------------------
def run_scanner(output_filename=OUTPUT_FILENAME, batch_size=BATCH_SIZE, verbose=True):
    symbols = fetch_nifty200_symbols()
    results = []

    for batch in tqdm(list(chunked(symbols, batch_size)), desc="Batches", ncols=120):
        tickers_str = " ".join(batch)
        raw = None
        try:
            raw = yf.download(tickers=tickers_str, period=YFINANCE_PERIOD, interval=YFINANCE_INTERVAL, group_by="ticker", progress=False)
        except Exception:
            raw = None

        tasks = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df_tk = raw[tk].dropna(how="all")
                        tasks.append(executor.submit(process_symbol_from_df, tk, df_tk))
                    except Exception:
                        tasks.append(executor.submit(process_symbol_download, tk))
            else:
                for tk in batch:
                    tasks.append(executor.submit(process_symbol_download, tk))

            for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Processing symbols", leave=False, ncols=120):
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        if verbose:
                            print(f"{res['Symbol']} → Net_Score={res['Net_Score']} Weighted={res['Weighted_Net_Score']} Summary={res['Summary']} CMP={res['CMP']} Change%={res['Change%']}")
                except Exception:
                    continue

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not results:
        print("No results collected.")
        return None

    final_df = pd.DataFrame(results, columns=["Symbol", "CMP", "Change%", "Net_Score", "Weighted_Net_Score", "Summary"])
    final_df = final_df.sort_values(by="Weighted_Net_Score", ascending=False)
    final_df.to_excel(output_filename, index=False)
    print(f"\nSaved: {output_filename}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    return final_df

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    df_final = run_scanner()
    if df_final is not None:
        print("\nTop 30 by Weighted_Net_Score:")
        print(df_final.head(30).to_string(index=False))

        strong_buy = df_final[df_final["Summary"] == "Strong Buy"].sort_values("Weighted_Net_Score", ascending=False)
        strong_sell = df_final[df_final["Summary"] == "Strong Sell"].sort_values("Weighted_Net_Score", ascending=False)

        print("\n" + "="*70)
        print("                 STRONG BUY STOCKS")
        print("="*70)
        if strong_buy.empty:
            print("No Strong Buy stocks at the moment.")
        else:
            print(strong_buy[["Symbol", "CMP", "Change%", "Weighted_Net_Score"]].to_string(index=False))

        print("\n" + "="*70)
        print("                 STRONG SELL STOCKS")
        print("="*70)
        if strong_sell.empty:
            print("No Strong Sell stocks at the moment.")
        else:
            print(strong_sell[["Symbol", "CMP", "Change%", "Weighted_Net_Score"]].to_string(index=False))

        print("\nScan completed.")

