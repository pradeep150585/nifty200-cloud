import subprocess
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
from common_functions import get_indices_summary
from common_functions import run_scanner, run_scanner_with_trend, process_symbol_from_df_with_volume
from common_functions import INDICES_FILE
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
# Consolidate All Outputs + Filter Final (Safe Version)
# ---------------------------------------
def consolidate_outputs(nifty_trend):
    print("\n📊 Consolidating all timeframe outputs...")

    # Timeframes for intraday analysis
    files = [FILE_1M, FILE_2M, FILE_5M, FILE_15M, FILE_30M]
    dfs = [safe_load(f) for f in files]

    # ✅ Early exit if any timeframe file missing or empty
    if any(df is None or df.empty for df in dfs):
        print("⚠ Some timeframe data missing. Skipping consolidation.")
        indices_summary = get_indices_summary(INDICES_FILE, interval="30m")
        return indices_summary, pd.DataFrame()

    df1, df2, df3, df4, df5 = dfs

    # --- Clean column names ---
    for df in dfs:
        df.columns = df.columns.astype(str).str.strip().str.replace("\u00A0", "", regex=False)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper()

    # --- Standardize and rename ---
    df1 = df1.rename(columns={"Summary": "Summary_1M"})[["Symbol", "Summary_1M"]]
    df2 = df2.rename(columns={"Summary": "Summary_2M"})[["Symbol", "Summary_2M"]]
    df3 = df3.rename(columns={"Summary": "Summary_5M", "CMP": "CMP_5M", "RSI": "RSI"})[
        ["Symbol", "CMP_5M", "Summary_5M", "RSI"]
    ]
    df4 = df4.rename(columns={"Summary": "Summary_15M"})[["Symbol", "Summary_15M"]]
    df5 = df5.rename(columns={"Summary": "Summary_30M", "Change%": "ChangePct", "VolumeRatio": "Volume"})[
        ["Symbol", "ChangePct", "Summary_30M", "Volume"]
    ]

    # --- Merge safely ---
    final = (
        df5.merge(df4, on="Symbol", how="left")
           .merge(df3, on="Symbol", how="left")
           .merge(df2, on="Symbol", how="left")
           .merge(df1, on="Symbol", how="left")
    )

    if final.empty:
        print("⚠ No data after merging — skipping.")
        indices_summary = get_indices_summary(INDICES_FILE, interval="30m")
        return indices_summary, pd.DataFrame()

    # --- Filter for consistent Strong Buy/Sell across all timeframes ---
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
    filtered = final.loc[mask, ["Symbol", "CMP_5M", "Summary_1M", "Summary_30M", "Volume", "RSI"]].rename(
        columns={"CMP_5M": "CMP", "Summary_1M": "Summary_Short", "Summary_30M": "Summary_Long"}
    )

    if filtered.empty:
        print("⚠ No stocks matching strong conditions.")
        indices_summary = get_indices_summary(INDICES_FILE, interval="30m")
        return indices_summary, pd.DataFrame()

    # --- Combine short + long term summary → Trend ---
    def combine_trend(row):
        if "Strong Buy" in row["Summary_Short"] and "Strong Buy" in row["Summary_Long"]:
            return "Strong Buy"
        elif "Strong Sell" in row["Summary_Short"] and "Strong Sell" in row["Summary_Long"]:
            return "Strong Sell"
        else:
            return "Neutral"

    filtered["Trend"] = filtered.apply(combine_trend, axis=1)

    # --- Volume Filter ---
    def parse_volume(v):
        try:
            return float(str(v).replace("x", ""))
        except:
            return 0.0

    filtered["VolumeValue"] = filtered["Volume"].apply(parse_volume)
    filtered = filtered[filtered["VolumeValue"] > 1.0]
    filtered = filtered.sort_values(by="VolumeValue", ascending=False).drop(columns=["VolumeValue"])

    # --- Indices Summary ---
    indices_summary = get_indices_summary(INDICES_FILE, interval="30m")
    if not indices_summary.empty:
        indices_summary = indices_summary[["Indices Name", "Trend", "RSI", "Change%"]]
        indices_summary = indices_summary.round(2)
        print("\n" + "="*80)
        print("📊 INDICES SUMMARY")
        print("="*80)
        print(indices_summary.to_string(index=False))
    else:
        print("\n⚠️ No indices data found or unable to fetch.")

    # --- Final Stock Table ---
    if not filtered.empty:
        if "RSI" in filtered.columns:
            filtered["RSI"] = filtered["RSI"].round(2)
        print("\n" + "="*80)
        print(f"📈 FINAL STRONG SIGNALS — NIFTY TREND: {nifty_trend.upper()}")
        print("="*80)
        display_cols = ["Symbol", "CMP", "Trend", "RSI"]
        if "Volume" in filtered.columns:
            display_cols.insert(3, "Volume")
        print(filtered[display_cols].to_string(index=False))
        filtered.to_excel(CONSOLIDATED_OUTPUT, index=False)
    else:
        print(f"\n⚠ No Strong Buy/Sell signals found. NIFTY TREND: {nifty_trend}\n")

    # ✅ Always return valid DataFrames
    return indices_summary, filtered

# ---------------------------------------
# Main Runner (Unified for Streamlit & Standalone)
# ---------------------------------------
def main(progress_callback=None, streamlit_mode=False):
    total_steps = 8
    current = 0

    def update():
        if progress_callback:
            progress_callback(int((current / total_steps) * 100))

    print("Starting Intraday Scanner Pipeline...\n")

    current += 1; update()

    # ---- STEP 1 ----
    nifty_trend, df_final = run_scanner_with_trend(
        period="60d", interval="30m",
        output_filename="Nifty200_Weighted_Balanced_30M_fixed.xlsx"
    )
    current += 1; update()

    # ---- STEP 2 ----
    run_scanner(period="14d", interval="15m",
                output_filename="Nifty200_Weighted_Balanced_15M_fixed.xlsx")
    current += 1; update()

    # ---- STEP 3 ----
    run_scanner(period="14d", interval="5m",
                output_filename="Nifty200_Weighted_Balanced_5M_fixed.xlsx")
    current += 1; update()

    # ---- STEP 4 ----
    run_scanner(period="2d", interval="2m",
                output_filename="Nifty200_Weighted_Balanced_2M_fixed.xlsx")
    current += 1; update()

    # ---- STEP 5 ----
    run_scanner(period="2d", interval="1m",
                output_filename="Nifty200_Weighted_Balanced_1M_fixed.xlsx")
    current += 1; update()

    # ---- STEP 6 ---- Load Nifty Trend ----
    try:
        if os.path.exists("Nifty_Trend.txt"):
            nifty_trend = open("Nifty_Trend.txt").read().strip() or "Neutral"
    except:
        nifty_trend = "Neutral"
    current += 1; update()

    # ---- STEP 7 ---- Final consolidation ----
    consolidate_outputs(nifty_trend)
    current += 1; update()

    if progress_callback:
        progress_callback(100)

    print("\n🌟 Intraday Scan Completed Successfully!\n")

    # --- Prepare Output Data ---
    indices_summary = get_indices_summary(INDICES_FILE, interval="30m")
    final_df = pd.read_excel("Nifty200_Consolidated_Output.xlsx") if os.path.exists("Nifty200_Consolidated_Output.xlsx") else pd.DataFrame()

    # --- Streamlit Mode ---
    if streamlit_mode:
        return indices_summary, final_df

    # --- Standalone Mode ---
    if not indices_summary.empty:
        print("\n" + "="*80)
        print("📊 INDICES SUMMARY")
        print("="*80)
        print(indices_summary[["Indices Name", "Trend", "RSI", "Change%"]].to_string(index=False))
    else:
        print("\n⚠️ No indices data found or unable to fetch.")

    if not final_df.empty:
        print("\n" + "="*80)
        print("📈 FINAL STRONG SIGNALS — INTRADAY")
        print("="*80)

        display_cols = ["Symbol", "CMP", "Trend", "RSI"]
        if "Volume" in final_df.columns:
            display_cols.insert(3, "Volume")

        final_df = final_df.round(2)
        print(final_df[display_cols].to_string(index=False))
    else:
        print("\n⚠️ No Strong Buy/Sell signals found.")

    return indices_summary, final_df


# ---------------------------------------
# Entry Point for Direct Run
# ---------------------------------------
if __name__ == "__main__":
    main()