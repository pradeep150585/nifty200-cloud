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
FILE_30M  = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"
FILE_1H  = "Nifty200_Weighted_Balanced_1H_fixed.xlsx"
FILE_4H  = "Nifty200_Weighted_Balanced_4H_fixed.xlsx"
FILE_1D = "Nifty200_Weighted_Balanced_1D_fixed.xlsx"
FILE_1W = "Nifty200_Weighted_Balanced_1W_fixed.xlsx"
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

    files = [FILE_30M, FILE_1H, FILE_4H, FILE_1D, FILE_1W]
    dfs = [safe_load(f) for f in files]
    if any(df.empty for df in dfs):
        print("⚠ Some timeframe data missing. Skipping consolidation.")
        return

    df1, df2, df3, df4, df5 = dfs

    for df in dfs:
        df.columns = df.columns.astype(str).str.strip().str.replace("\u00A0", "", regex=False)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper()

    df1 = df1.rename(columns={"Summary": "Summary_30M"})[["Symbol", "Summary_30M"]]
    df2 = df2.rename(columns={"Summary": "Summary_1H"})[["Symbol", "Summary_1H"]]
    df3 = df3.rename(columns={"Summary": "Summary_4H", "CMP": "CMP_4H"})[["Symbol", "CMP_4H", "Summary_4H"]]
    df4 = df4.rename(columns={"Summary": "Summary_1D"})[["Symbol", "Summary_1D"]]
    df5 = df5.rename(columns={"Summary": "Summary_1W", "Change%": "ChangePct", "VolumeRatio": "Volume"})[
        ["Symbol", "ChangePct", "Summary_1W", "Volume"]
    ]

    final = (
        df5.merge(df1, on="Symbol", how="left")
             .merge(df2, on="Symbol", how="left")
             .merge(df3, on="Symbol", how="left")
             .merge(df4, on="Symbol", how="left")
    )

    # Keep only Strong Buy/Sell combinations
    def strong_condition(x):
        summaries = x[["Summary_30M", "Summary_1H", "Summary_4H", "Summary_1D", "Summary_1W"]]
        if summaries.isna().any():
            return False
        if all(v == "Strong Buy" for v in summaries):
            return True
        if all(v == "Strong Sell" for v in summaries):
            return True
        return False

    mask = final.apply(strong_condition, axis=1)
    filtered = final.loc[mask, ["Symbol", "CMP_4H", "Summary_30M", "Summary_1W", "Volume"]].rename(
        columns={"CMP_4H": "CMP", "Summary_30M": "Summary_Medium", "Summary_1W": "Summary_Long"}
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
def main():
    print("Starting Auto Scanner Pipeline...\n")
    nifty_trend, df_final = run_scanner_with_trend(period="5y", interval="1wk", output_filename="Nifty200_Weighted_Balanced_1W_fixed.xlsx")
    run_scanner(period="2y", interval="1d", output_filename="Nifty200_Weighted_Balanced_1D_fixed.xlsx")
    run_scanner(period="1y", interval="4h", output_filename="Nifty200_Weighted_Balanced_4H_fixed.xlsx")
    run_scanner(period="60d", interval="1h", output_filename="Nifty200_Weighted_Balanced_1H_fixed.xlsx")
    run_scanner(period="60d", interval="30m", output_filename="Nifty200_Weighted_Balanced_30M_fixed.xlsx")

    # Read trend
    nifty_trend = "Neutral"
    try:
        if os.path.exists("Nifty_Trend.txt"):
            nifty_trend = open("Nifty_Trend.txt").read().strip() or "Neutral"
    except Exception:
        nifty_trend = "Neutral"

    consolidate_outputs(nifty_trend)
    print("\n🌟 All tasks completed successfully!\n")


if __name__ == "__main__":
    main()