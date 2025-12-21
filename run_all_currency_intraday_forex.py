import os
import time
import pandas as pd
from common_functions import run_scanner, safe_load

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FILE_15M = "Currency_15M.xlsx"
FILE_30M = "Currency_30M.xlsx"

# ---------------------------------------------------------------------
# Utility: wait for file (Streamlit-safe)
# ---------------------------------------------------------------------
def wait_for_file(path, timeout=20):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            return False
        time.sleep(0.5)
    return True

# ---------------------------------------------------------------------
# Consolidation Logic
# ---------------------------------------------------------------------
def consolidate_outputs():
    df_15 = safe_load(FILE_15M)
    df_30 = safe_load(FILE_30M)

    if df_15.empty or df_30.empty:
        return pd.DataFrame()

    df_15 = df_15.rename(columns={"Summary": "Summary_15M"})[["Symbol", "Summary_15M"]]

    df_30 = df_30.rename(columns={
        "Summary": "Summary_30M",
        "RSI": "RSI",
        "ADX": "ADX",
        "CMP": "CMP",
        "Change%": "Change%"
    })

    # Required columns (must exist)
    required_cols = ["Symbol", "CMP", "Summary_30M", "RSI", "Change%"]

    # Optional columns (may or may not exist)
    optional_cols = ["ADX"]

    # Select only available columns
    cols = required_cols + [c for c in optional_cols if c in df_30.columns]
    df_30 = df_30[cols]

    final = df_30.merge(df_15, on="Symbol", how="left")

    if "ADX" in final.columns and final["ADX"].isna().all():
        final = final.drop(columns=["ADX"])

    def trend_logic(row):
        if row["Summary_15M"] == "Strong Buy" and row["Summary_30M"] == "Strong Buy":
            return "Strong Buy"
        if row["Summary_15M"] == "Strong Sell" and row["Summary_30M"] == "Strong Sell":
            return "Strong Sell"
        return "Neutral"

    final["Trend"] = final.apply(trend_logic, axis=1)

    final = final[final["Trend"].isin(["Strong Buy", "Strong Sell"])]
    ui_cols = ["Symbol", "CMP", "Trend", "RSI"]
    if "ADX" in final.columns:
        ui_cols.append("ADX")
    ui_cols.append("Change%")

    final = final[ui_cols]
    final["RSI"] = final["RSI"].round(2)
    if "ADX" in final.columns:
        final["ADX"] = final["ADX"].round(2)
    final["CMP"] = final["CMP"].round(5)

    sort_cols = ["RSI"]
    if "ADX" in final.columns:
        sort_cols.append("ADX")

    final = final.sort_values(by=sort_cols, ascending=[False]*len(sort_cols))

    return final.reset_index(drop=True)

# ---------------------------------------------------------------------
# Main Runner (Streamlit + Standalone compatible)
# ---------------------------------------------------------------------
def main(progress_callback=None, streamlit_mode=False):
    # ---- 15M ----
    if progress_callback:
        progress_callback(25)

    run_scanner(
        period="60d",
        interval="15m",
        output_filename=FILE_15M,
        symbol_source="forex"
    )

    if not wait_for_file(FILE_15M):
        return pd.DataFrame()

    # ✅ 15M completed
    if progress_callback:
        progress_callback(50)

    # ---- 30M ----
    run_scanner(
        period="60d",
        interval="30m",
        output_filename=FILE_30M,
        symbol_source="forex"
    )

    if not wait_for_file(FILE_30M):
        return pd.DataFrame()

    # ✅ 30M completed
    if progress_callback:
        progress_callback(75)

    # ---- Consolidation ----
    final_df = consolidate_outputs()

    if progress_callback:
        progress_callback(100)

    if streamlit_mode:
        return final_df

    if not final_df.empty:
        print("\n📈 FINAL STRONG SIGNALS — FOREX INTRADAY\n")
        print(final_df.to_string(index=False))

    return final_df

if __name__ == "__main__":
    main()