import streamlit as st
import subprocess
import time
import pandas as pd
import threading
import os
import os
APP_DIR = os.path.dirname(os.path.abspath(__file__))
INDICES_FILE_STREAMLIT = os.path.join(APP_DIR, "indices.txt")
from common_functions import fetch_nifty200_symbols
from run_all_intraday import main as run_intraday_main
from run_all_swing import main as run_swing_main
from run_all_currency_intraday_forex import main as run_forex_main

# ==========================
# Global Output Management
# ==========================
def get_main_container():
    """Get or create the main output container."""
    if "main_output" not in st.session_state:
        st.session_state["main_output"] = st.container()
    return st.session_state["main_output"]

def clear_main_container():
    """Completely clears the main output area before a new scan."""
    if "main_output" in st.session_state:
        st.session_state["main_output"].empty()
        st.session_state["main_output"] = st.container()

# --- Create or reset global container ---
def reset_output_container():
    """Fully clears all dynamic UI elements (progress, banners, tables)."""
    if "output_container" not in st.session_state:
        st.session_state["output_container"] = st.empty()
    else:
        st.session_state["output_container"].empty()
        st.session_state["output_container"] = st.empty()
    return st.session_state["output_container"]

# ---------- Utility: Clear all Streamlit elements ----------
def clear_all_streamlit_outputs():
    # Clears progress, text, banners, and tables
    for placeholder in st.session_state.get("active_placeholders", []):
        try:
            placeholder.empty()
        except Exception:
            pass
    st.session_state["active_placeholders"] = []
    st.empty()

# ---------- Utility: Track Streamlit placeholders ----------
def register_placeholder(p):
    """Keeps track of placeholders for easy clearing."""
    if "active_placeholders" not in st.session_state:
        st.session_state["active_placeholders"] = []
    st.session_state["active_placeholders"].append(p)
    return p

# ---------- Page Setup ----------
st.set_page_config(page_title="ScanBot AI", layout="wide")
# ---------- Persistent Footer (always visible) ----------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #222;
        color: gray;
        text-align: center;
        padding: 10px 0;
        font-size: 0.9em;
        border-top: 1px solid #444;
        z-index: 99999;
    }
    </style>
    <div class="footer">Developed by <b>Pradeep Kumar Palani</b></div>
""", unsafe_allow_html=True)
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}  /* Hides the top-right menu */
    footer {visibility: hidden;}     /* Hides “Made with Streamlit” */
    header {visibility: hidden;}     /* Optional: hides the header */
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)
# ---------- CSS ----------
st.markdown("""
<style>
    /* Top spacing */
    .block-container {
        padding-top: 10px !important;
    }

    /* Import Darkly CSS */
    @import url('https://bootswatch.com/5/darkly/bootstrap.min.css');

    /* Background Alignment */
    body, .stApp, .main {
        background-color: #222 !important;
        color: #fff !important;
    }

    /* Title */
    .main-heading {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 0 0 6px 0;
        line-height: 1.1;
    }

    .sub-heading {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #adb5bd;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
        margin: 0 0 20px 0;
    }

    /* Buttons */
    .button-container { display:flex; justify-content:center; gap:24px; margin-bottom:22px; }
    .stButton>button {
        display:inline-block;
        min-width:180px;
        padding:8px 16px;
        font-size:16px;
        font-weight:600;
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        border-radius:6px;

        background-color:#375a7f !important;
        border:1px solid #2b4763 !important;
        color:#ffffff !important;
    }
    .stButton>button:hover {
        background-color:#2b4763 !important;
        border-color:#1f3347 !important;
        color:#ffffff !important;
    }

    /* Table Container */
    .table-container {
        width:100%;
        margin:20px 0;
        padding:0;
        background:#2b2b2b;
        border-radius:10px;
        box-shadow:0 4px 12px rgba(0,0,0,0.35);
        overflow:hidden;
    }

    .table-container table {
        width:100% !important;
        border-collapse:collapse;
    }

    .table-container thead th {
        background:#375a7f !important;
        color:#fff !important;
        font-weight:600;
        border-bottom:1px solid rgba(255,255,255,0.15);
        padding:12px 14px;
        position:sticky;
        top:0;
        z-index:5;
    }

    /* Make each row carry its own background so hover can work */
    .table-container tbody tr {
        background-color: #ffffff;
    }

    .table-container tbody td {
        padding:12px 14px;
        vertical-align:middle;
        border-bottom:1px solid #ddd;
        /* IMPORTANT: make cell background transparent so tr:hover shows */
        background: transparent !important;
    }

    /* Alternating stripe (rows) */
    .table-container tbody tr:nth-child(odd):not(.success-row):not(.danger-row) {
        background-color:#fafafa !important;
    }

    /* Hover: change the cells' background when the row is hovered */
    .table-container tbody tr:hover td {
        background-color:#e6e6e6 !important;
    }

    /* Success / Danger row colors (matching Darkly palette) */
    .success-row td { background-color:#1abc9c !important; color:#ffffff !important; font-weight:600; }
    .danger-row td { background-color:#e74c3c !important; color:#ffffff !important; font-weight:600; }

    /* Mobile Responsive */
    @media (max-width: 640px) {
        .block-container {
            padding-left: 8px !important;
            padding-right: 8px !important;
        }
        .table-container table {
            font-size: 0.78rem !important;
        }
        .main-heading {
            font-size: 1.8rem !important;
        }
        .sub-heading {
            font-size: 0.85rem !important;
        }
        .stButton>button {
            width: 100% !important;
            min-width: 100% !important;
            margin-bottom: 10px;
        }
    }

    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color:#375a7f !important;
        border-radius:4px;
        height: 20px;
    }

    .no-border-table th, .no-border-table td { border:none !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='main-heading'>ScanBot AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-heading'>Nifty 200 AI-powered Analysis Dashboard</div>", unsafe_allow_html=True)

# ---------- Script Runner ----------
def run_script_with_progress(parent, script, excel_out):
    with parent:
        progress_bar = register_placeholder(st.progress(0))
        progress_text = register_placeholder(st.empty())

    progress_value = 0

    # Callback used by scanner scripts
    def progress_callback(percent):
        nonlocal progress_value
        progress_value = percent
        progress_bar.progress(percent)
        progress_text.markdown(f"<div style='text-align:center;'>Running... {percent}%</div>", unsafe_allow_html=True)

    try:
        if script == "run_all_intraday.py":
            indices_df, final_df = run_intraday_main(progress_callback, streamlit_mode=True, indices_file=INDICES_FILE_STREAMLIT)
        else:
            indices_df, final_df = run_swing_main(progress_callback, streamlit_mode=True, indices_file=INDICES_FILE_STREAMLIT)

        # ---- Normalize Indices Trend labels (UI layer) ----
        if isinstance(indices_df, pd.DataFrame) and not indices_df.empty:
            if "Trend" in indices_df.columns:
                indices_df["Trend"] = indices_df["Trend"].replace({
                    # Strong signals
                    "Strong Bullish": "Strong Buy",
                    "Bullish": "Strong Buy",
                    "Strong Bearish": "Strong Sell",
                    "Bearish": "Strong Sell",

                    # Mild signals
                    "Mild Bullish": "Buy",
                    "Mild Bearish": "Sell"
                })

        # ---- Format Indices numeric columns (2 decimals) ----
        if isinstance(indices_df, pd.DataFrame) and not indices_df.empty:
            for col in ["RSI", "ADX", "Change%"]:
                if col in indices_df.columns:
                    indices_df[col] = pd.to_numeric(
                        indices_df[col], errors="coerce"
                    ).round(2)

        # ---- Sort Indices: RSI ↓ then ADX ↓ ----
        if isinstance(indices_df, pd.DataFrame) and not indices_df.empty:
            sort_cols = [c for c in ["RSI", "ADX"] if c in indices_df.columns]
            if sort_cols:
                indices_df = indices_df.sort_values(
                    by=sort_cols,
                    ascending=[False] * len(sort_cols)
                )

        # --- Display Indices Summary Table ---
        if isinstance(indices_df, pd.DataFrame) and not indices_df.empty:
            st.markdown("<div style='text-align:left; font-size:1.3rem; margin:10px 0;'><b>📊 Indices Summary</b></div>", unsafe_allow_html=True)

            def highlight_trend(row):
                color = "black"
                if "Bullish" in row["Trend"]:
                    color = "green"
                elif "Bearish" in row["Trend"]:
                    color = "red"
                return [f"color: {color}; font-weight: bold;" if col == "Trend" else "" for col in row.index]

            def style_indices_row(r):
                if r["Trend"] == "Strong Buy":
                    return ['color:#1abc9c;font-weight:400;'] * len(r)
                elif r["Trend"] == "Buy":
                    return ['color:#27ae60;font-weight:400;'] * len(r)
                elif r["Trend"] == "Strong Sell":
                    return ['color:#e74c3c;font-weight:400;'] * len(r)
                elif r["Trend"] == "Sell":
                    return ['color:#c0392b;font-weight:400;'] * len(r)
                else:
                    return ['color:#000000;font-weight:400;'] * len(r)

            display_cols = indices_df.columns.tolist()

            if "ADX" not in display_cols and "ADX" in indices_df.columns:
                display_cols.insert(display_cols.index("RSI") + 1, "ADX")

            indices_df = indices_df[display_cols]

            styled_indices = (
                indices_df
                .style
                .apply(style_indices_row, axis=1)
                .format({
                    "RSI": "{:.2f}",
                    "ADX": "{:.2f}",
                    "Change%": "{:.2f}"
                })
            )

            if hasattr(styled_indices, "hide_index"):
                styled_indices = styled_indices.hide_index()
            else:
                styled_indices = styled_indices.hide(axis="index")

            indices_html = styled_indices.to_html(
                classes="table table-hover table-dark no-border-table"
            )

            st.markdown(
                f"<div class='table-container'>{indices_html}</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠ No indices summary data found.")

        progress_bar.progress(100)
        progress_text.markdown("<div style='text-align:center;'>Completed. Loading results…</div>", unsafe_allow_html=True)
        time.sleep(1)

    except Exception as e:
        progress_text.markdown(f"<div style='text-align:center;color:red;'>Error: {e}</div>", unsafe_allow_html=True)
        return

    output_area = st.container()
    with output_area:
        if os.path.exists(excel_out):
            try:
                df = pd.read_excel(excel_out)

                if not df.empty:
                    # ✅ Clean column names to remove hidden spaces or case mismatches
                    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)

                    # ---- Final Trend calculation ----
                    df["Trend"] = "Neutral"

                    if all(c in df.columns for c in ["Summary_Short", "Summary_Long"]):
                        strong_buy = (
                            (df["Summary_Short"] == "Strong Buy") &
                            (df["Summary_Long"] == "Strong Buy")
                        )

                        strong_sell = (
                            (df["Summary_Short"] == "Strong Sell") &
                            (df["Summary_Long"] == "Strong Sell")
                        )

                        df.loc[strong_buy, "Trend"] = "Strong Buy"
                        df.loc[strong_sell, "Trend"] = "Strong Sell"

                    # ✅ Combine both conditions — keep any stock satisfying either condition
                    df = df.copy()
                    if df.empty:
                        st.warning("No stocks matched final UI conditions.")
                        return

                    # ---- Remove unwanted columns from Stock Summary ----
                    for col in ["Summary_Short", "Summary_Long"]:
                        if col in df.columns:
                            df.drop(columns=col, inplace=True)

                    if "CMP" in df.columns:
                        df["CMP"] = df["CMP"].astype(float).round(2)

                    def style_row(r):
                        if r.get("Trend") == "Strong Buy":
                            return ['color:#1abc9c;font-weight:400;'] * len(r)
                        elif r.get("Trend") == "Strong Sell":
                            return ['color:#e74c3c;font-weight:400;'] * len(r)
                        elif r.get("Trend") == "Neutral":
                            return ['color:#000000;font-weight:400;'] * len(r)
                        return [''] * len(r)

                    try:
                        with open("Nifty_Trend.txt", "r") as f:
                            nifty_value = f.read().strip()
                        st.markdown(f"<div style='text-align:left; font-size:1.2rem; margin:10px 0;'><b>Nifty Trend:</b> {nifty_value}</div>", unsafe_allow_html=True)
                    except:
                        st.markdown("<div style='text-align:left; font-size:1.2rem; margin:10px 0; color:#bbb;'><b>Nifty Trend:</b> Not Available</div>", unsafe_allow_html=True)

                    for col in ["RSI", "ADX"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

                    styled = (
                        df.style
                        .apply(style_row, axis=1)
                        .format({
                            "CMP": "{:.2f}",
                            "RSI": "{:.2f}",
                            "ADX": "{:.2f}",
                            "Change%": "{:.2f}"
                        })
                    )

                    if hasattr(styled, "hide_index"):
                        styled = styled.hide_index()
                    else:
                        styled = styled.hide(axis="index")

                    styled_html = styled.to_html(classes="table table-hover no-border-table")
                    st.markdown(f"<div class='table-container'>{styled_html}</div>", unsafe_allow_html=True)

                else:
                    st.warning("No strong signals found.")

            except Exception as e:
                st.error(f"Error reading output file: {e}")

        else:
            st.error("Output file not found.")

def run_forex_with_progress(parent):
    progress_bar = register_placeholder(st.progress(0))
    progress_text = register_placeholder(st.empty())

    def progress_callback(p):
        progress_bar.progress(p)

        # ✅ Same text style as Intraday/Swing
        if p < 100:
            progress_text.markdown(
                f"<div style='text-align:center;'>Running... {p}%</div>",
                unsafe_allow_html=True
            )
        else:
            progress_text.markdown(
                "<div style='text-align:center;'>Completed. Loading results…</div>",
                unsafe_allow_html=True
            )

    # ---- Run Forex Scanner ----
    final_df = run_forex_main(
        progress_callback=progress_callback,
        streamlit_mode=True
    )

    if final_df is None or final_df.empty:
        st.warning("⚠ No Strong Buy/Sell signals found in Forex/Crypto.")
        return

    # ✅ Truncate RSI & ADX
    for col in ["RSI", "ADX"]:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").round(2)

    def style_row(r):
        if r["Trend"] == "Strong Buy":
            return ['color:#1abc9c;font-weight:400;'] * len(r)
        elif r["Trend"] == "Strong Sell":
            return ['color:#e74c3c;font-weight:400;'] * len(r)
        return [''] * len(r)

    styled = (
        final_df.style
        .apply(style_row, axis=1)
        .format({
            "CMP": "{:.2f}",
            "RSI": "{:.2f}",
            "ADX": "{:.2f}",
            "Change%": "{:.2f}"
        })
    )

    styled = styled.hide(axis="index") if hasattr(styled, "hide") else styled.hide_index()

    st.markdown(
        "<div style='font-size:1.3rem; margin:10px 0;'><b>📈 Forex / Crypto Strong Signals</b></div>",
        unsafe_allow_html=True
    )

    html = styled.to_html(classes="table table-hover no-border-table")
    st.markdown(f"<div class='table-container'>{html}</div>", unsafe_allow_html=True)

# ---------- Buttons ----------
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

run_intraday = col1.button("Run Intraday Scanner", use_container_width=True)
run_swing = col2.button("Run Swing Scanner", use_container_width=True)
run_forex = col3.button("Run Forex / Crypto Scanner", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

output_placeholder = st.container()

if run_intraday or run_swing or run_forex:
    clear_main_container()
    output_placeholder = get_main_container()
    with output_placeholder:        

        try:
            if run_intraday or run_swing:
            # Step 1: Always download latest Nifty 200 list
                st.info("🌐 Downloading latest Nifty 200 stock list from NSE...")
                fetch_nifty200_symbols()
                st.success("✅ Nifty 200 list downloaded successfully!")

            # Step 2: Run selected scanner
            if run_intraday:
                st.info("🚀 Running Intraday Scanner...")
                run_script_with_progress(output_placeholder, "run_all_intraday.py", "Nifty200_Consolidated_Output.xlsx")
            elif run_swing:
                st.info("📊 Running Swing Scanner...")
                run_script_with_progress(output_placeholder, "run_all_swing.py", "Nifty200_Consolidated_Output.xlsx")
            elif run_forex:
                st.info("💱 Running Forex / Crypto Scanner...")
                run_forex_with_progress(output_placeholder)

        except SystemExit as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")