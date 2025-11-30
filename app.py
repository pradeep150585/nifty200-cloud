# app.py
import streamlit as st
import subprocess, os, sys, time, random
from pathlib import Path
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="Nifty 200 AI Scanner", layout="wide")

# ------------------------------  MATERIAL UI (GOOGLE STYLE)  -------------------------------- #
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #f8f9fa;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --muted: #64748b;
    --primary: #1a73e8;
    --primary-hover: #1664c4;
    --radius: 10px;
}

html, body, .main, .block-container {
    background: var(--bg);
    color: var(--text);
    font-family: 'Roboto', sans-serif;
    margin: 0;
    padding: 0;
}

/*************** NAVBAR (Fixed Top) ***************/
.navbar {
    width: 100%;
    background: white;
    border-bottom: 1px solid var(--border);
    position: fixed;
    top:0;
    left:0;
    z-index: 9999;
    padding: 14px 28px;
    box-sizing: border-box;
}
.nav-title {
    font-size: 22px;
    font-weight: 500;
}
.nav-subtitle {
    margin-top: -2px;
    font-size: 13.5px;
    color: var(--muted);
}

/* Content padding so navbar does not overlap */
.block-container {
    padding-top: 120px !important;
}

/*************** CARD ***************/
.card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 24px;
    border: 1px solid var(--border);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}

/*************** BUTTONS (Material Style) ***************/
.stButton>button {
    width: 100%;
    background: var(--primary) !important;
    color: white !important;
    border-radius: 6px;
    padding: 10px 18px;
    font-size: 15px;
    font-weight: 500;
    border: none;
    transition: all .18s ease-in-out;
}
.stButton>button:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
}

/*************** PROGRESS BAR ***************/
.progress-bar {
  width: 100%;
  height: 10px;
  border-radius: var(--radius);
  background: #e4e7eb;
  margin-top: 18px;
}
.progress-fill {
  height: 10px;
  border-radius: var(--radius);
  transition: width .25s ease;
  background: var(--primary);
}

/*************** CONSOLE ***************/
pre, code, .stCode {
    width: 100% !important;
    max-width: 100% !important;
}
pre {
    background: #f1f3f4 !important;
    padding: 18px !important;
    border-radius: var(--radius) !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    border: 1px solid #e0e0e0 !important;
    overflow-x: auto !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------  CENTERED HEADER SECTION  ------------------------------------ #
# Use components.html to reliably render the centered header (avoids escaping)
header_html = """
<div style="width:100%; text-align:center; margin-top: 6px; padding: 10px 0 20px 0;">
  <h1 style="font-size: 32px; font-weight: 500; color: #1e293b; margin: 0 0 6px 0;">
    NIFTY 200 AI Scanner
  </h1>
  <p style="font-size: 15px; color: #64748b; font-weight: 400; margin: 0;">
    AI-driven market scanner — Intraday & Swing Trade modes
  </p>
</div>
"""
# components.html will render HTML exactly and not be escaped; small height avoids huge iframe
components.html(header_html, height=110)

# ------------------------------------------------------------------------------------------- #
#                           SCANNER FUNCTION (UNCHANGED)                                      #
# ------------------------------------------------------------------------------------------- #
def run_scanner(script, label):
    if not Path(script).exists():
        st.error(f"❌ {script} not found.")
        return

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, script]

    anim = st.empty()
    prog = st.empty()
    logs_box = st.empty()

    frames = ["[AI]","[AI..]","[AI...]","[AI↺]"]
    colors = ["#1a73e8","#1664c4","#10b981"]
    progress, i = 0, 0
    logs = ""

    with st.spinner(f"Running {label}..."):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env
        )

        for line in iter(proc.stdout.readline, ''):
            if not line:
                break

            frame = frames[i % len(frames)]
            color = colors[(progress // 34) % len(colors)]
            progress = min(progress + random.randint(1,3), 100)

            # Animation panel (kept as a card)
            anim.markdown(
                f"<div class='card' style='text-align:center;'><strong>{frame} {label} — scanning market data</strong></div>",
                unsafe_allow_html=True
            )

            # Progress bar
            prog.markdown(f"""
                <div class='progress-bar'>
                    <div class='progress-fill' style='width:{progress}%;background:{color};'></div>
                </div>
            """, unsafe_allow_html=True)

            # Append logs (limit lines)
            logs += line
            if len(logs.splitlines()) > 400:
                logs = "\n".join(logs.splitlines()[-400:])

            # Render logs unchanged using st.code (full width via CSS above)
            logs_box.code(logs, language="bash")

            time.sleep(0.25)
            i += 1

        proc.wait()

    # Finalize progress
    prog.markdown("""
        <div class='progress-bar'>
            <div class='progress-fill' style='width:100%;background:#10b981;'></div>
        </div>
    """, unsafe_allow_html=True)

    anim.markdown(
        "<div class='card' style='text-align:center;'><strong>✅ Scan Complete — AI insights ready</strong></div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------------------- #
#                           BUTTON CARD (Material UI)                                         #
# ------------------------------------------------------------------------------------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    b1, b2 = st.columns(2)
    start_intraday = b1.button("Intraday Scanner")
    start_swing = b2.button("Swing Trade Scanner")

st.markdown('</div>', unsafe_allow_html=True)

# Full-width output below card
if start_intraday:
    run_scanner("run_all_intraday.py", "Intraday Scanner")

if start_swing:
    run_scanner("run_all_swing.py", "Swing Trade Scanner")
