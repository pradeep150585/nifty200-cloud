import subprocess
import os
import pandas as pd
from tqdm import tqdm
from time import sleep

# -----------------------------
# CONFIG: Output filenames
# -----------------------------
FILE_5M  = "Nifty200_Weighted_Balanced_5M_fixed.xlsx"
FILE_15M = "Nifty200_Weighted_Balanced_15M_fixed.xlsx"
FILE_30M = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"
FILE_1D  = "Nifty200_Weighted_Balanced_1D_fixed.xlsx"

CONSOLIDATED_OUTPUT = "Nifty200_Consolidated_Output.xlsx"

# -----------------------------
# RUN A SCRIPT WITH PROGRESS BAR
# -----------------------------
def run_script(script_name):
    import subprocess

    print(f"\n🔥 Running: {script_name} ...\n")

    process = subprocess.Popen(
        ["python", "-X", "utf8", script_name],
        stdout=subprocess.DEVNULL,       # Don't read stdout (avoids blocking)
        stderr=subprocess.PIPE           # Capture raw bytes
    )

    # Read stderr safely without UnicodeDecodeError
    stderr_bytes = process.stderr.read()
    stderr = stderr_bytes.decode("utf-8", errors="ignore")

    process.wait()

    if process.returncode != 0:
        print(f"❌ Error in {script_name}:\n{stderr}\n")
    else:
        print(f"✅ {script_name} completed successfully.\n")

# -----------------------------
# SAFE LOAD EXCEL
# -----------------------------
def safe_load(path):
    if not os.path.exists(path):
        print(f"⚠ Missing file: {path}")
        return pd.DataFrame()
    return pd.read_excel(path)

# -----------------------------
# MAIN MERGE FUNCTION
# -----------------------------
def consolidate_outputs():

    print("\n📊 Consolidating timeframe outputs...")

    # Progress bar for loading files  
    files = [FILE_5M, FILE_15M, FILE_30M, FILE_1D]
    dfs = []

    for f in tqdm(files, desc="Loading Excel files", ncols=90):
        dfs.append(safe_load(f))

    df5, df15, df30, df1d = dfs

    # -----------------------------------------------------------
    # NORMALIZE ALL COLUMN NAMES (fix space, NBSP, hidden chars)
    # -----------------------------------------------------------
    for df in dfs:
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.replace("\u00A0", "", regex=False)
        )

        # Normalize symbols
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.upper()

    # -----------------------------------------------------------
    # 5M FRAME
    # -----------------------------------------------------------
    df5 = df5.rename(columns={
        "CMP": "CMP_5M",
        "Weighted_Net_Score": "NetScore_5M",
        "Summary": "Summary_5M"
    })

    df5 = df5[["Symbol", "CMP_5M", "NetScore_5M", "Summary_5M"]]

    # -----------------------------------------------------------
    # 15M FRAME
    # -----------------------------------------------------------
    df15 = df15.rename(columns={
        "Weighted_Net_Score": "NetScore_15M",
        "Summary": "Summary_15M"
    })

    df15 = df15[["Symbol", "NetScore_15M", "Summary_15M"]]

    # -----------------------------------------------------------
    # 30M FRAME
    # -----------------------------------------------------------
    df30 = df30.rename(columns={
        "Weighted_Net_Score": "NetScore_30M",
        "Summary": "Summary_30M"
    })

    df30 = df30[["Symbol", "NetScore_30M", "Summary_30M"]]

    # -----------------------------------------------------------
    # DAILY FRAME (REAL COLUMN NAMES FROM YOUR SCREENSHOT)
    # -----------------------------------------------------------
    df1d = df1d.rename(columns={
        "Change%": "ChangePct",
        "Weighted_Net_Score": "NetScore_1D",
        "Summary": "Summary_1D"
    })

    df1d = df1d[["Symbol", "ChangePct", "NetScore_1D", "Summary_1D"]]

    # -----------------------------------------------------------
    # MERGE ALL TIMEFRAMES
    # -----------------------------------------------------------
    print("\n🔄 Merging all timeframes...")

    final = (
        df1d.merge(df5,  on="Symbol", how="left")
             .merge(df15, on="Symbol", how="left")
             .merge(df30, on="Symbol", how="left")
    )

    # -----------------------------------------------------------
    # FINAL COLUMN ORDER
    # -----------------------------------------------------------
    final = final[
        [
            "Symbol",
            "CMP_5M",
            "ChangePct",
            "NetScore_5M",  "Summary_5M",
            "NetScore_15M", "Summary_15M",
            "NetScore_30M", "Summary_30M",
            "NetScore_1D",  "Summary_1D",
        ]
    ]

    # Save file
    if os.path.exists(CONSOLIDATED_OUTPUT):
        os.remove(CONSOLIDATED_OUTPUT)

    final.to_excel(CONSOLIDATED_OUTPUT, index=False)

    # -----------------------------------------------------------
    # SHOW ONLY STRONG BUY / STRONG SELL IN ALL 4 TIMEFRAMES
    # -----------------------------------------------------------
    mask = final[["Summary_5M", "Summary_15M", "Summary_30M", "Summary_1D"]].apply(
        lambda x: all(v in ["Strong Buy", "Strong Sell"] for v in x), axis=1
    )

    filtered = final.loc[mask, [
        "Symbol", "CMP_5M", "ChangePct",
        "Summary_5M", "Summary_15M", "Summary_30M", "Summary_1D"
    ]]

    print(f"\n🎉 Final consolidated file created → {CONSOLIDATED_OUTPUT}\n")

    if not filtered.empty:
        print("📈 Stocks that are Strong Buy / Strong Sell in all 5M, 15M, 30M, and 1D:\n")
        print(filtered.to_string(index=False))
    else:
        print("⚠️ No stocks found that are Strong Buy / Strong Sell across all 4 timeframes.\n")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Auto Scanner Pipeline...\n")

    run_script("Daily.py")
    run_script("30M.py")
    run_script("15M.py")
    run_script("5M.py")

    consolidate_outputs()

    print("\n🌟 All tasks completed successfully!\n")

