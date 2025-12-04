import subprocess
import os
import pandas as pd
import sys
from tqdm import tqdm

FILE_1M  = "Nifty200_Weighted_Balanced_1M_fixed.xlsx"
FILE_2M  = "Nifty200_Weighted_Balanced_2M_fixed.xlsx"
FILE_5M  = "Nifty200_Weighted_Balanced_5M_fixed.xlsx"
FILE_15M = "Nifty200_Weighted_Balanced_15M_fixed.xlsx"
FILE_30M = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"
CONSOLIDATED_OUTPUT = "Nifty200_Consolidated_Output.xlsx"

def run_script(script_name):
    print(f"\n🔥 Running: {script_name} ...\n")
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    out, err = process.communicate()
    if process.returncode != 0:
        print(f"❌ Error in {script_name}:\n{err}\n")
    else:
        print(out)
        print(f"✅ {script_name} completed successfully.\n")

def safe_load(path):
    if not os.path.exists(path):
        print(f"⚠ Missing file: {path}")
        return pd.DataFrame()
    return pd.read_excel(path)

def consolidate_outputs(nifty_trend):
    print("\n📊 Consolidating timeframe outputs...")

    files = [FILE_1M, FILE_2M, FILE_5M, FILE_15M, FILE_30M]
    dfs = [safe_load(f) for f in files]
    df1, df2, df5, df15, df30 = dfs

    for df in dfs:
        df.columns = df.columns.astype(str).str.strip().str.replace("\u00A0", "", regex=False)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper()

    df1 = df1.rename(columns={"Summary": "Summary_1M"})[["Symbol", "Summary_1M"]]
    df2 = df2.rename(columns={"Summary": "Summary_2M"})[["Symbol", "Summary_2M"]]
    df5 = df5.rename(columns={"Summary": "Summary_5M", "CMP": "CMP_5M"})[["Symbol", "CMP_5M", "Summary_5M"]]
    df15 = df15.rename(columns={"Summary": "Summary_15M"})[["Symbol", "Summary_15M"]]
    df30 = df30.rename(columns={"Summary": "Summary_30M", "Change%": "ChangePct", "VolumeRatio": "Volume"})[
        ["Symbol", "ChangePct", "Summary_30M", "Volume"]
    ]

    final = (
        df30.merge(df1, on="Symbol", how="left")
             .merge(df2, on="Symbol", how="left")
             .merge(df5, on="Symbol", how="left")
             .merge(df15, on="Symbol", how="left")
    )

    def strong_condition(x):
        summaries = x[["Summary_1M", "Summary_2M", "Summary_5M", "Summary_15M", "Summary_30M"]]
        if summaries.isna().any():
            return False
        if all(v == "Strong Buy" for v in summaries):
            return True
        if all(v == "Strong Sell" for v in summaries):
            return True
        return False

    mask = final.apply(strong_condition, axis=1)
    filtered = final.loc[mask, ["Symbol", "CMP_5M", "ChangePct", "Summary_1M", "Summary_30M", "Volume"]].rename(
        columns={"CMP_5M": "CMP", "ChangePct": "Change%", "Summary_1M": "Summary_Medium", "Summary_30M": "Summary_Long"}
    )

    # Convert volume to numeric multiplier (e.g., "1.5x" -> 1.5) for sorting
    def parse_volume(v):
        try:
            return float(str(v).replace("x", ""))
        except:
            return 0.0

    filtered["VolumeValue"] = filtered["Volume"].apply(parse_volume)
    filtered = filtered.sort_values(by="VolumeValue", ascending=False).drop(columns=["VolumeValue"])

    if not filtered.empty:
        print("\n" + "="*85)
        print(f"📈 FINAL STRONG SIGNALS — NIFTY TREND: {nifty_trend.upper()}")
        print("="*85)
        print(filtered.to_string(index=False))
    else:
        print(f"\n⚠️ No Strong Buy/Sell signals. NIFTY TREND: {nifty_trend}\n")

if __name__ == "__main__":
    print("🚀 Starting Auto Scanner Pipeline...\n")

    # --- Run scripts in order (30M first to get Nifty Trend) ---
    run_script("30M.py")

    # Extract Nifty Trend from 30M.py log file (if printed)
    nifty_trend = "Neutral"
    try:
        # Search last run’s trend in a text file if saved, else default
        if os.path.exists("Nifty_Trend.txt"):
            nifty_trend = open("Nifty_Trend.txt").read().strip()
    except Exception:
        nifty_trend = "Neutral"

    run_script("15M.py")
    run_script("5M.py")
    run_script("2M.py")
    run_script("1M.py")

    consolidate_outputs(nifty_trend)
    print("\n🌟 All tasks completed successfully!\n")