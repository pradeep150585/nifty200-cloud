import subprocess
import os
import pandas as pd
from tqdm import tqdm
from time import sleep

# -----------------------------
# CONFIG: Output filenames
# -----------------------------
FILE_1M  = "Nifty200_Weighted_Balanced_1M_fixed.xlsx"
FILE_2M  = "Nifty200_Weighted_Balanced_2M_fixed.xlsx"
FILE_5M  = "Nifty200_Weighted_Balanced_5M_fixed.xlsx"
FILE_15M = "Nifty200_Weighted_Balanced_15M_fixed.xlsx"
FILE_30M = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"

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
    files = [FILE_1M, FILE_2M, FILE_5M, FILE_15M, FILE_30M]
    dfs = []

    for f in tqdm(files, desc="Loading Excel files", ncols=90):
        dfs.append(safe_load(f))

    df1, df2, df5, df15, df30 = dfs

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
    # 1M FRAME
    # -----------------------------------------------------------
    df1 = df1.rename(columns={
        "CMP": "CMP_1M",
        "Weighted_Net_Score": "NetScore_1M",
        "Summary": "Summary_1M"
    })

    df1 = df1[["Symbol", "NetScore_1M", "Summary_1M"]]

    # -----------------------------------------------------------
    # 1M FRAME
    # -----------------------------------------------------------
    df2 = df2.rename(columns={
        "CMP": "CMP_2M",
        "Weighted_Net_Score": "NetScore_2M",
        "Summary": "Summary_2M"
    })

    df2 = df2[["Symbol", "NetScore_2M", "Summary_2M"]]

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
        "CMP": "CMP_15M",
        "Weighted_Net_Score": "NetScore_15M",
        "Summary": "Summary_15M"
    })

    df15 = df15[["Symbol", "NetScore_15M", "Summary_15M"]]

    # -----------------------------------------------------------
    # 30M FRAME
    # -----------------------------------------------------------
    df30 = df30.rename(columns={
        "CMP": "CMP_30M",
        "Change%": "ChangePct",
        "Weighted_Net_Score": "NetScore_30M",
        "Summary": "Summary_30M"
    })

    df30 = df30[["Symbol", "ChangePct", "NetScore_30M", "Summary_30M"]]

    # -----------------------------------------------------------
    # MERGE ALL TIMEFRAMES
    # -----------------------------------------------------------
    print("\n🔄 Merging all timeframes...")

    final = (
        df30.merge(df1,  on="Symbol", how="left")
             .merge(df2,  on="Symbol", how="left")
             .merge(df5, on="Symbol", how="left")
             .merge(df15, on="Symbol", how="left")
    )

    # -----------------------------------------------------------
    # FINAL COLUMN ORDER
    # -----------------------------------------------------------
    final = final[
        [
            "Symbol",
            "CMP_5M",
            "ChangePct",
            "NetScore_1M",  "Summary_1M",
            "NetScore_1M",  "Summary_2M",
            "NetScore_5M",  "Summary_5M",
            "NetScore_15M", "Summary_15M",
            "NetScore_30M", "Summary_30M",
         ]
    ]

    # Save file
    if os.path.exists(CONSOLIDATED_OUTPUT):
        os.remove(CONSOLIDATED_OUTPUT)

    final.to_excel(CONSOLIDATED_OUTPUT, index=False)

    # -----------------------------------------------------------
    # SHOW ONLY STRONG BUY / STRONG SELL IN ALL 5 TIMEFRAMES
    # -----------------------------------------------------------
    def buy_sell_condition(x):
        # Buy condition
        if all(v == "Strong Buy" for v in x[["Summary_1M", "Summary_2M", "Summary_5M"]]) and \
           all(v in ["Strong Buy", "Buy"] for v in x[["Summary_15M", "Summary_30M"]]):
            return True

        # Sell condition
        if all(v == "Strong Sell" for v in x[["Summary_1M", "Summary_2M", "Summary_5M"]]) and \
           all(v in ["Strong Sell", "Sell"] for v in x[["Summary_15M", "Summary_30M"]]):
            return True

        return False  # ✅ must be inside the function

    mask = final.apply(buy_sell_condition, axis=1)

    filtered = final.loc[mask, ["Symbol", "CMP_5M", "ChangePct", "Summary_1M", "Summary_30M"]].rename(
        columns={
            "CMP_5M": "CMP",
            "ChangePct": "Change%_30M",
            "Summary_1M": "Summary_Medium",
            "Summary_30M": "Summary_Long"
        }
    )

    print(f"\n🎉 Final consolidated file created → {CONSOLIDATED_OUTPUT}\n")

    if not filtered.empty:
        print("📈 Stocks matching Strong Buy / Strong Sell logic across all timeframes:\n")
        print(filtered.to_string(index=False))
    else:
        print("⚠️ No stocks found matching the criteria.\n")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Auto Scanner Pipeline...\n")

    #run_script("30M.py")
    #run_script("15M.py")
    #run_script("5M.py")
    #run_script("2M.py")
    #run_script("1M.py")

    consolidate_outputs()

    print("\n🌟 All tasks completed successfully!\n")