import io
import math
import time
import requests
import os
from datetime import datetime
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------
# CONFIG
# -----------------------
NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
MOM_PERIOD = 10
MIN_ROWS_REQUIRED = 80
MAX_WORKERS = 6
SLEEP_BETWEEN_BATCHES = 0.5

# -----------------------
# UTILITIES
# -----------------------
def fetch_nifty200_symbols() -> List[str]:
    """
    Always download the Nifty 200 stock list freshly from NSE API.
    No cache, no fallback — if it fails, raise an exception.
    """
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        print("🌐 Downloading Nifty 200 list from NSE...")
        session = requests.Session()
        # NSE requires an initial request to set cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        if "data" not in data:
            raise ValueError("Unexpected response format from NSE API")

        df = pd.DataFrame(data["data"])
        df.to_csv("ind_nifty200list.csv", index=False)
        symbols = df["symbol"].astype(str).str.strip().tolist()
        symbols = [s for s in symbols if s.isalpha() and len(s) <= 10]
        print(f"✅ Successfully downloaded {len(symbols)} Nifty 200 symbols from NSE")
        return [s.upper() + ".NS" for s in symbols]

    except Exception as e:
        print(f"❌ Failed to download Nifty 200 list: {e}")
        raise SystemExit("Stopping execution — Unable to fetch Nifty 200 list from NSE.")

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


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
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))

# -----------------------
# INDICATORS (Investing.com-style)
# -----------------------
def compute_indicators_vectorized(df: pd.DataFrame, mom_period=MOM_PERIOD) -> pd.DataFrame:
    if not {"Close", "High", "Low"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Close', 'High', 'Low' columns.")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI
    period_rsi = 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period_rsi)
    avg_loss = wilder_smooth(loss, period_rsi)
    RS = safe_div(avg_gain, avg_loss)
    df["RSI"] = 100 - (100 / (1 + RS))
    df["RSI_Rating"] = df["RSI"].apply(lambda v: "Buy" if v > 60 else ("Sell" if v < 30 else "Neutral"))

    # Stochastic Oscillator (9,3,3)
    period_stoch = 9
    low9 = low.rolling(period_stoch, min_periods=period_stoch).min()
    high9 = high.rolling(period_stoch, min_periods=period_stoch).max()
    k_raw = safe_div((close - low9), (high9 - low9)) * 100
    k_smooth = k_raw.rolling(3, min_periods=3).mean()
    d_smooth = k_smooth.rolling(3, min_periods=3).mean()
    df["STO_Rating"] = k_smooth.apply(lambda v: "Buy" if v < 20 else ("Sell" if v > 80 else "Neutral"))

    # StochRSI
    stochrsi_period = 14
    rsi_min = df["RSI"].rolling(stochrsi_period, min_periods=stochrsi_period).min()
    rsi_max = df["RSI"].rolling(stochrsi_period, min_periods=stochrsi_period).max()
    stochrsi = safe_div(df["RSI"] - rsi_min, (rsi_max - rsi_min)) * 100
    df["StochRSI_Rating"] = stochrsi.apply(lambda v: "Buy" if v < 20 else ("Sell" if v > 80 else "Neutral"))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Rating"] = "Neutral"
    df.loc[cross_up(macd, macd_signal), "MACD_Rating"] = "Buy"
    df.loc[cross_down(macd, macd_signal), "MACD_Rating"] = "Sell"

    # True Range & ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    TR = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = wilder_smooth(TR, 14)

    # ADX
    up_move = high.diff()
    down_move = -low.diff()
    DMp = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    DMm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    DMp_w = wilder_smooth(DMp, 14)
    DMm_w = wilder_smooth(DMm, 14)
    DIp = 100 * safe_div(DMp_w, df["ATR"])
    DIm = 100 * safe_div(DMm_w, df["ATR"])
    DX = 100 * safe_div((DIp - DIm).abs(), (DIp + DIm))
    ADX = wilder_smooth(DX, 14)
    df["ADX_Rating"] = ["Buy" if (a > 30 and b > c) else ("Sell" if (a > 30 and c > b) else "Neutral")
                        for a, b, c in zip(ADX, DIp, DIm)]

    # Williams %R
    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    df["WR"] = -100 * safe_div(hh14 - close, (hh14 - ll14))
    df["WR_Rating"] = df["WR"].apply(lambda v: "Buy" if v < -80 else ("Sell" if v > -20 else "Neutral"))

    # CCI
    tp = (high + low + close) / 3.0
    tp_sma = tp.rolling(14).mean()
    mad = (tp - tp_sma).abs().rolling(14).mean()
    df["CCI"] = safe_div((tp - tp_sma), (0.015 * mad))
    df["CCI_Rating"] = df["CCI"].apply(lambda v: "Buy" if v < -100 else ("Sell" if v > 100 else "Neutral"))

    # ROC
    df["ROC"] = safe_div((close - close.shift(12)), close.shift(12)) * 100
    df["ROC_Rating"] = df["ROC"].apply(lambda v: "Buy" if v > 0 else ("Sell" if v < 0 else "Neutral"))

    # SMA/EMA
    for m in [5, 10, 20, 50, 100]:
        df[f"SMA{m}"] = close.rolling(m).mean()
        df[f"EMA{m}"] = close.ewm(span=m, adjust=False).mean()
        df[f"SMA{m}_Rating"] = np.where(df["Close"] > df[f"SMA{m}"], "Buy", "Sell")
        df[f"EMA{m}_Rating"] = np.where(df["Close"] > df[f"EMA{m}"], "Buy", "Sell")

    # Momentum
    df[f"MOM_{mom_period}"] = close.diff(mom_period)
    df[f"MOM_{mom_period}_Rating"] = df[f"MOM_{mom_period}"].apply(lambda v: "Buy" if v > 0 else ("Sell" if v < 0 else "Neutral"))

    # Summary
    rating_cols = [c for c in df.columns if c.endswith("_Rating")]
    df_ratings = df[rating_cols].replace({"Buy": 1, "Sell": -1, "Neutral": 0})
    osc_cols = [
        "RSI_Rating", "STO_Rating", "StochRSI_Rating", "CCI_Rating",
        "WR_Rating", "ROC_Rating", "MACD_Rating", "ADX_Rating", f"MOM_{mom_period}_Rating"
    ]
    ma_cols = [f"SMA{m}_Rating" for m in [5, 10, 20, 50, 100]] + [f"EMA{m}_Rating" for m in [5, 10, 20, 50, 100]]
    osc_ratio = (df_ratings[osc_cols] == 1).sum(axis=1) - (df_ratings[osc_cols] == -1).sum(axis=1)
    ma_ratio = (df_ratings[ma_cols] == 1).sum(axis=1) - (df_ratings[ma_cols] == -1).sum(axis=1)
    df["Oscillator_Buy%"] = osc_ratio / len(osc_cols)
    df["MA_Buy%"] = ma_ratio / len(ma_cols)

    def classify_summary(row):
        o = row["Oscillator_Buy%"]
        m = row["MA_Buy%"]
        combined = 0.7 * m + 0.3 * o
        if m > 0.55 and o > 0.45:
            combined += 0.1
        if combined >= 0.55:
            return "Strong Buy"
        elif combined >= 0.15:
            return "Buy"
        elif combined <= -0.55:
            return "Strong Sell"
        elif combined <= -0.15:
            return "Sell"
        else:
            return "Neutral"

    df["Summary"] = df.apply(classify_summary, axis=1)
    df["Weighted_Net_Score"] = df_ratings.sum(axis=1)
    df["Net_Score"] = df_ratings.sum(axis=1)
    return df

# -----------------------
# PROCESSING
# -----------------------
def process_symbol_from_df(symbol: str, df_tk: pd.DataFrame):
    try:
        if df_tk.empty or len(df_tk) < MIN_ROWS_REQUIRED:
            return None
        df_tk = df_tk.sort_index()
        df_ind = compute_indicators_vectorized(df_tk.copy())
        last = df_ind.iloc[-1]
        cmp_price = float(last["Close"])
        prev_close = float(df_ind["Close"].iloc[-2]) if len(df_ind) >= 2 else np.nan
        change_pct = round((cmp_price - prev_close) / prev_close * 100, 2) if prev_close and not math.isnan(prev_close) else None
        return {
            "Symbol": symbol.replace(".NS", ""),
            "CMP": round(cmp_price, 2),
            "Change%": change_pct,
            "Net_Score": int(last["Net_Score"]),
            "Weighted_Net_Score": int(last["Weighted_Net_Score"]),
            "Summary": last["Summary"],
            "RSI": round(float(last.get("RSI", np.nan)), 2) if "RSI" in df_ind.columns else np.nan
        }
    except Exception:
        return None


def process_symbol_download(symbol: str, period: str, interval: str):
    try:
        df_tk = yf.download(symbol, period=period, interval=interval, progress=False)
        if df_tk.empty or len(df_tk) < MIN_ROWS_REQUIRED:
            return None
        return process_symbol_from_df(symbol, df_tk)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

def process_symbol_download_with_volume(symbol, period, interval):
    try:
        df_tk = yf.download(symbol, period=period, interval=interval, progress=False)
        if df_tk.empty or len(df_tk) < MIN_ROWS_REQUIRED:
            return None
        return process_symbol_from_df_with_volume(symbol, df_tk)
    except Exception:
        return None

# -----------------------
# RUN SCANNER
# -----------------------
def run_scanner(period, interval, output_filename, batch_size=20, verbose=True):
    symbols = fetch_nifty200_symbols()
    results = []

    for batch in tqdm(list(chunked(symbols, batch_size)), desc="Batches", ncols=120):
        tickers_str = " ".join(batch)
        try:
            raw = yf.download(tickers=tickers_str, period=period, interval=interval,
                              group_by="ticker", progress=False)
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
                        tasks.append(executor.submit(process_symbol_download, tk, period, interval))
            else:
                for tk in batch:
                    tasks.append(executor.submit(process_symbol_download, tk, period, interval))

            for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Processing symbols", leave=False, ncols=120):
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        if verbose:
                            print(f"{res['Symbol']} → Net_Score={res['Net_Score']} Weighted={res['Weighted_Net_Score']} "
                                  f"Summary={res['Summary']} CMP={res['CMP']} Change%={res['Change%']}")
                except Exception:
                    continue

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not results:
        print("No results collected.")
        return None

    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by="Weighted_Net_Score", ascending=False)
    final_df.to_excel(output_filename, index=False)
    print(f"\nSaved: {output_filename}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    return final_df

# -----------------------
# NIFTY TREND ANALYSIS
# -----------------------
def get_nifty_trend_summary(interval="30m"):
    """Detect NIFTY trend using the same indicator logic as for stocks."""
    import pandas as pd
    import yfinance as yf

    print(f"\n📊 Fetching NIFTY {interval} data for trend detection...")
    df = yf.download("^NSEI", period="60d", interval=interval, progress=False, auto_adjust=False)

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).strip().capitalize() for c in df.columns]

    if not {"Close", "High", "Low"}.issubset(df.columns):
        print(f"⚠️ Missing required columns in Nifty data: {list(df.columns)}")
        return "Neutral"

    try:
        df = compute_indicators_vectorized(df)
    except Exception as e:
        print(f"⚠️ Indicator computation failed: {e}")
        return "Neutral"

    if "Summary" not in df.columns or df.empty:
        print("⚠️ Missing Summary column in indicator results.")
        return "Neutral"

    last_summary = str(df.iloc[-1]["Summary"])

    if "Strong Buy" in last_summary:
        trend = "Bullish"
    elif "Strong Sell" in last_summary:
        trend = "Bearish"
    elif "Buy" in last_summary:
        trend = "Mild Bullish"
    elif "Sell" in last_summary:
        trend = "Mild Bearish"
    else:
        trend = "Neutral"

    print(f"📈 NIFTY SUMMARY: {last_summary} → TREND: {trend}")
    return trend

# -----------------------
# INDEX SUMMARY TABLE (Final Optimized Version)
# -----------------------
def get_indices_summary(file_path, interval="30m"):
    """
    Read indices.txt formatted as:
    Bank - ^NSEBANK
    Nifty - ^NSEI
    FinServ - ^CNXFINANCE

    For each index, compute Trend, RSI, and Change% using the given interval.
    """

    import os

    if not os.path.exists(file_path):
        print(f"⚠️ Missing indices file: {file_path}")
        return pd.DataFrame()

    # Read "Name - Symbol" pairs
    pairs = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "-" not in line:
                continue
            name, symbol = [x.strip() for x in line.split("-", 1)]
            pairs.append((name, symbol))

    if not pairs:
        print("⚠️ No valid index entries found in indices.txt")
        return pd.DataFrame()

    results = []

    for name, symbol in pairs:
        try:
            df = yf.download(symbol, period="30d", interval=interval, progress=False)

            # Flatten MultiIndex and normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.columns = [str(c).capitalize().strip() for c in df.columns]

            # Ensure essential columns exist
            if not {"Close", "High", "Low"}.issubset(df.columns):
                print(f"⚠️ Skipping {name} ({symbol}) — missing required OHLC columns.")
                continue

            if df.empty or len(df) < 10:
                print(f"⚠️ Skipping {name} ({symbol}) — insufficient data.")
                continue

            # Compute indicators
            df = compute_indicators_vectorized(df)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            cmp_price = float(last["Close"])
            prev_price = float(prev["Close"])
            change_pct = round((cmp_price - prev_price) / prev_price * 100, 2)
            rsi = round(float(last.get("RSI", np.nan)), 2)

            summary = last.get("Summary", "Neutral")

            # Simplified trend mapping
            if "Strong Buy" in summary:
                trend = "Bullish"
            elif "Strong Sell" in summary:
                trend = "Bearish"
            elif "Buy" in summary:
                trend = "Mild Bullish"
            elif "Sell" in summary:
                trend = "Mild Bearish"
            else:
                trend = "Neutral"

            results.append({
                "Indices Name": name,
                "Trend": trend,
                "RSI": rsi,
                "Change%": change_pct
            })

        except Exception as e:
            print(f"⚠️ Error processing {name} ({symbol}): {e}")
            continue

    if not results:
        print("⚠️ No valid index data processed.")
        return pd.DataFrame()

    df_final = pd.DataFrame(results)
    df_final = df_final.round(2)

    # Sort by RSI descending
    df_final = df_final.sort_values(by="RSI", ascending=False).reset_index(drop=True)
    return df_final

# -----------------------
# ENHANCED PROCESSING (with volume ratio)
# -----------------------
def process_symbol_from_df_with_volume(symbol: str, df_tk: pd.DataFrame):
    base = process_symbol_from_df(symbol, df_tk)
    if base is None:
        return None

    try:
        avg_vol20 = df_tk["Volume"].tail(20).mean()
        last_vol = df_tk["Volume"].iloc[-1]
        vol_ratio = round(last_vol / avg_vol20, 2) if avg_vol20 and avg_vol20 > 0 else np.nan
        base["VolumeRatio"] = f"{vol_ratio}x" if not np.isnan(vol_ratio) else "N/A"
    except Exception:
        base["VolumeRatio"] = "N/A"

    return base


# -----------------------
# COMBINED TREND + SCANNER
# -----------------------
def run_scanner_with_trend(period, interval, output_filename, batch_size=20, verbose=True):
    nifty_trend = get_nifty_trend_summary(interval)
    with open("Nifty_Trend.txt", "w") as f:
        f.write(nifty_trend)
    print(f"\n💾 Saved Nifty Trend: {nifty_trend}")

    # For 30m – use volume-enhanced scanner
    if interval in ["30m", "1wk"]:
        df_final = run_scanner_with_volume(period, interval, output_filename, batch_size, verbose)
    else:
        df_final = run_scanner(period, interval, output_filename, batch_size, verbose)

    return nifty_trend, df_final

def run_scanner_with_volume(period, interval, output_filename, batch_size=20, verbose=True):
    symbols = fetch_nifty200_symbols()
    results = []

    for batch in tqdm(list(chunked(symbols, batch_size)), desc="Batches", ncols=120):
        tickers_str = " ".join(batch)
        try:
            raw = yf.download(
                tickers=tickers_str,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False
            )
        except Exception:
            raw = None

        tasks = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        df_tk = raw[tk].dropna(how="all")
                        tasks.append(executor.submit(process_symbol_from_df_with_volume, tk, df_tk))
                    except Exception:
                        tasks.append(executor.submit(process_symbol_download_with_volume, tk, period, interval))
            else:
                for tk in batch:
                    tasks.append(executor.submit(process_symbol_download_with_volume, tk, period, interval))

            for fut in tqdm(as_completed(tasks), total=len(tasks),
                            desc="Processing symbols", leave=False, ncols=120):
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        if verbose:
                            print(
                                f"{res['Symbol']} → Net_Score={res['Net_Score']} "
                                f"Weighted={res['Weighted_Net_Score']} "
                                f"Summary={res['Summary']} CMP={res['CMP']} "
                                f"Change%={res['Change%']} VolumeRatio={res.get('VolumeRatio')}"
                            )
                except Exception:
                    continue

        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not results:
        print("No results collected.")
        return None

    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by="Weighted_Net_Score", ascending=False)
    final_df.to_excel(output_filename, index=False)
    print(f"\nSaved: {output_filename} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    return final_df