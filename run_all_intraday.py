import subprocess
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
from common_functions import run_scanner, run_scanner_with_trend, process_symbol_from_df_with_volume
import sys

# Disable tqdm progress bars if launched from Streamlit
if "--no-tqdm" in sys.argv or "--safe" in sys.argv:
    from tqdm import tqdm
    tqdm.__init__ = lambda *args, **kwargs: iter([])

# ---------------------------------------
# File Paths
# ---------------------------------------
FILE_1M  = "Nifty200_Weighted_Balanced_1M_fixed.xlsx"
FILE_2M  = "Nifty200_Weighted_Balanced_2M_fixed.xlsx"
FILE_5M  = "Nifty200_Weighted_Balanced_5M_fixed.xlsx"
FILE_15M = "Nifty200_Weighted_Balanced_15M_fixed.xlsx"
FILE_30M = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"
CONSOLIDATED_OUTPUT = "Nifty200_Consolidated_Output.xlsx"


# ---------------------------------------
# Helper: Run Sub-Scripts Silently
# ---------------------------------------
def run_script(script_name):
    process = subprocess.Popen(
        [sys.executable, "-X", "utf8", script_name],
        stdout=subprocess.DEVNULL,   # Silent output
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )
    _, err = process.communicate()
    if process.returncode != 0:
        print(f"❌ Error in {script_name}: {err}")


# ---------------------------------------
# Helper: Load Excel Safely
# ---------------------------------------
def safe_load(path):
    if not os.path.exists(path):
        print(f"⚠ Missing file: {path}")
        return pd.DataFrame()
    return pd.read_excel(path)


# ---------------------------------------
# Consolidate All Outputs + Filter Final
# ---------------------------------------
def consolidate_outputs(nifty_trend):
    print("\n📊 Consolidating all timeframe outputs...")

    files = [FILE_1M, FILE_2M, FILE_5M, FILE_15M, FILE_30M]
    dfs = [safe_load(f) for f in files]
    if any(df.empty for df in dfs):
        print("⚠ Some timeframe data missing. Skipping consolidation.")
        return

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

    # Keep only Strong Buy/Sell combinations
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
    filtered = final.loc[mask, ["Symbol", "CMP_5M", "Summary_1M", "Summary_30M", "Volume"]].rename(
        columns={"CMP_5M": "CMP", "Summary_1M": "Summary_Medium", "Summary_30M": "Summary_Long"}
    )

    # Merge Medium + Long Summary into single "Trend" column
    def combine_trend(row):
        if "Strong Buy" in row["Summary_Medium"] and "Strong Buy" in row["Summary_Long"]:
            return "Strong Buy"
        elif "Strong Sell" in row["Summary_Medium"] and "Strong Sell" in row["Summary_Long"]:
            return "Strong Sell"
        else:
            return "Neutral"

    filtered["Trend"] = filtered.apply(combine_trend, axis=1)

    # Sort by Volume descending
    def parse_volume(v):
        try:
            return float(str(v).replace("x", ""))
        except:
            return 0.0

    filtered["VolumeValue"] = filtered["Volume"].apply(parse_volume)
    filtered = filtered.sort_values(by="VolumeValue", ascending=False).drop(columns=["VolumeValue"])

    # --- Final Display ---
    if not filtered.empty:
        print("\n" + "="*80)
        print(f"📈 FINAL STRONG SIGNALS — NIFTY TREND: {nifty_trend.upper()}")
        print("="*80)
        print(filtered[["Symbol", "CMP", "Trend", "Volume"]].to_string(index=False))
        filtered.to_excel(CONSOLIDATED_OUTPUT, index=False)
    else:
        print(f"\n⚠️ No Strong Buy/Sell signals. NIFTY TREND: {nifty_trend}\n")


# ---------------------------------------
# Main Runner
# ---------------------------------------
if __name__ == "__main__":
    print("Starting Auto Scanner Pipeline...\n")
    nifty_trend, df_final = run_scanner_with_trend(period="60d", interval="30m", output_filename="Nifty200_Weighted_Balanced_30M_fixed.xlsx")
    run_scanner(period="14d", interval="15m", output_filename="Nifty200_Weighted_Balanced_15M_fixed.xlsx")
    run_scanner(period="14d", interval="5m", output_filename="Nifty200_Weighted_Balanced_5M_fixed.xlsx")
    run_scanner(period="2d", interval="2m", output_filename="Nifty200_Weighted_Balanced_2M_fixed.xlsx")
    run_scanner(period="2d", interval="1m", output_filename="Nifty200_Weighted_Balanced_1M_fixed.xlsx")
 
    # Read Nifty Trend saved by 30M.py
    nifty_trend = "Neutral"
    try:
        if os.path.exists("Nifty_Trend.txt"):
            nifty_trend = open("Nifty_Trend.txt").read().strip() or "Neutral"
    except Exception:
        nifty_trend = "Neutral"

    consolidate_outputs(nifty_trend)
    print("\n🌟 All tasks completed successfully!\n")