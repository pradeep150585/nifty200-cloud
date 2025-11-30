import subprocess
import os
import pandas as pd
from tqdm import tqdm
from time import sleep

# -----------------------------
# CONFIG: Output filenames
# -----------------------------
FILE_30M  = "Nifty200_Weighted_Balanced_30M_fixed.xlsx"
FILE_1H  = "Nifty200_Weighted_Balanced_1H_fixed.xlsx"
FILE_4H  = "Nifty200_Weighted_Balanced_4H_fixed.xlsx"
FILE_1D = "Nifty200_Weighted_Balanced_1D_fixed.xlsx"
FILE_1W = "Nifty200_Weighted_Balanced_1W_fixed.xlsx"

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
    files = [FILE_30M, FILE_1H, FILE_4H, FILE_1D, FILE_1W]
    dfs = []

    for f in tqdm(files, desc="Loading Excel files", ncols=90):
        dfs.append(safe_load(f))

    df1, df2, df3, df4, df5 = dfs

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
    # 30M FRAME
    # -----------------------------------------------------------
    df1 = df1.rename(columns={
        "CMP": "CMP_30M",
        "Weighted_Net_Score": "NetScore_30M",
        "Summary": "Summary_30M"
    })

    df1 = df1[["Symbol", "CMP_30M", "NetScore_30M", "Summary_30M"]]

    # -----------------------------------------------------------
    # 1H FRAME
    # -----------------------------------------------------------
    df2 = df2.rename(columns={
        "CMP": "CMP_1H",
        "Weighted_Net_Score": "NetScore_1H",
        "Summary": "Summary_1H"
    })

    df2 = df2[["Symbol", "NetScore_1H", "Summary_1H"]]

    # -----------------------------------------------------------
    # 4H FRAME
    # -----------------------------------------------------------
    df3 = df3.rename(columns={
        "CMP": "CMP_4H",
        "Weighted_Net_Score": "NetScore_4H",
        "Summary": "Summary_4H"
    })

    df3 = df3[["Symbol", "NetScore_4H", "Summary_4H"]]

    # -----------------------------------------------------------
    # 1D FRAME
    # -----------------------------------------------------------
    df4 = df4.rename(columns={
        "CMP": "CMP_1D",
        "Change%": "ChangePct",
        "Weighted_Net_Score": "NetScore_1D",
        "Summary": "Summary_1D"
    })

    df4 = df4[["Symbol", "ChangePct", "NetScore_1D", "Summary_1D"]]

    # -----------------------------------------------------------
    # 1W FRAME
    # -----------------------------------------------------------
    df5 = df5.rename(columns={
        "CMP": "CMP_1W",
        "Weighted_Net_Score": "NetScore_1W",
        "Summary": "Summary_1W"
    })

    df5 = df5[["Symbol", "NetScore_1W", "Summary_1W"]]

    # -----------------------------------------------------------
    # MERGE ALL TIMEFRAMES
    # -----------------------------------------------------------
    print("\n🔄 Merging all timeframes...")

    final = (
        df5.merge(df1,  on="Symbol", how="left")
             .merge(df2,  on="Symbol", how="left")
             .merge(df3, on="Symbol", how="left")
             .merge(df4, on="Symbol", how="left")
    )

    # -----------------------------------------------------------
    # FINAL COLUMN ORDER
    # -----------------------------------------------------------
    final = final[
        [
            "Symbol",
            "CMP_30M",
            "ChangePct",
            "NetScore_30M",  "Summary_30M",
            "NetScore_1H",  "Summary_1H",
            "NetScore_4H",  "Summary_4H",
            "NetScore_1D", "Summary_1D",
            "NetScore_1W", "Summary_1W",
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
        if all(v == "Strong Buy" for v in x[["Summary_30M", "Summary_1H", "Summary_4H"]]) and \
           all(v in ["Strong Buy", "Buy"] for v in x[["Summary_1D", "Summary_1W"]]):
            return True

        # Sell condition
        if all(v == "Strong Sell" for v in x[["Summary_30M", "Summary_1H", "Summary_4H"]]) and \
           all(v in ["Strong Sell", "Sell"] for v in x[["Summary_1D", "Summary_1W"]]):
            return True

        return False

    mask = final.apply(buy_sell_condition, axis=1)

    filtered = final.loc[mask, ["Symbol", "CMP_30M", "ChangePct", "Summary_30M", "Summary_1W"]].rename(
        columns={
            "CMP_30M": "CMP",
            "ChangePct": "Change%_1D",
            "Summary_30M": "Summary_Medium",
            "Summary_1W": "Summary_Long"
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

    #run_script("1W.py")
    #run_script("1D.py")
    #run_script("4H.py")
    #run_script("1H.py")
    #run_script("30M.py")

    consolidate_outputs()

    print("\n🌟 All tasks completed successfully!\n")