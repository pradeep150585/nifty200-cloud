import streamlit as st
import subprocess
import time
import pandas as pd
import threading
import os
from run_all_intraday import main as run_intraday_main
from run_all_swing import main as run_swing_main


# ---------- Page Setup ----------
st.set_page_config(page_title="ScanBot AI", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
    /* Top spacing */
    .block-container {
        padding-top: 80px !important;
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
def run_script_with_progress(script, excel_out):
    progress_area = st.container()
    with progress_area:
        progress_bar = st.progress(0)
        progress_text = st.empty()

    progress_value = 0

    # Callback used by scanner scripts
    def progress_callback(percent):
        nonlocal progress_value
        progress_value = percent
        progress_bar.progress(percent)
        progress_text.markdown(f"<div style='text-align:center;'>Running... {percent}%</div>", unsafe_allow_html=True)

    try:
        if script == "run_all_intraday.py":
            run_intraday_main(progress_callback)
        else:
            run_swing_main(progress_callback)

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
                    if "CMP" in df.columns:
                        df["CMP"] = df["CMP"].astype(float).round(2)

                    def style_row(r):
                        if isinstance(r.get("Trend"), str):
                            if "Strong Buy" in r["Trend"]:
                                return ['background-color:#eafaf6;color:#1abc9c;font-weight:400;']*len(r)
                            elif "Strong Sell" in r["Trend"]:
                                return ['background-color:#fdecea;color:#e74c3c;font-weight:400;']*len(r)
                        return ['']*len(r)

                    try:
                        with open("Nifty_Trend.txt", "r") as f:
                            nifty_value = f.read().strip()
                        st.markdown(f"<div style='text-align:left; font-size:1.2rem; margin:10px 0;'><b>Nifty Trend:</b> {nifty_value}</div>", unsafe_allow_html=True)
                    except:
                        st.markdown("<div style='text-align:left; font-size:1.2rem; margin:10px 0; color:#bbb;'><b>Nifty Trend:</b> Not Available</div>", unsafe_allow_html=True)
 
                    styled = df.style.apply(style_row, axis=1).format({"CMP": "{:.2f}"})

                    if hasattr(styled, "hide_index"):
                        styled = styled.hide_index()
                    else:
                        styled = styled.hide(axis="index")
                    styled_html = styled.to_html(classes="table table-hover table-dark no-border-table")
                    st.markdown(f"<div class='table-container'>{styled_html}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No strong signals found.")
            except Exception as e:
                st.error(f"Error reading output file: {e}")
        else:
            st.error("Output file not found.")

# ---------- Buttons ----------
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])
run_intraday = col1.button("Run Intraday Scanner", use_container_width=True)
run_swing = col2.button("Run Swing Scanner", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_intraday:
    run_script_with_progress("run_all_intraday.py", "Nifty200_Consolidated_Output.xlsx")
elif run_swing:
    run_script_with_progress("run_all_swing.py", "Nifty200_Consolidated_Output.xlsx")

# ---------- Footer ----------
st.markdown("""
<hr>
<div style="text-align:center;color:gray;font-size:0.9em;">
Developed by <b>Pradeep Kumar Palani</b>
</div>
""", unsafe_allow_html=True)
