# app.py
import streamlit as st
import subprocess
import os
import sys
from pathlib import Path

# Streamlit page configuration
st.set_page_config(page_title="Nifty200 Scanner", layout="wide")
st.title("📊 NIFTY 200 Technical Scanner (Streamlit Community Cloud)")
st.write("Click **Run Full Scanner** to run your pipeline. Output is streamed below.")

# --------------------------------------------------------------------
# Safety: verify all required scripts exist
# --------------------------------------------------------------------
required = ["run_all.py", "5M.py", "15M.py", "30M.py", "Daily.py"]
missing = [f for f in required if not Path(f).exists()]
if missing:
    st.error("Missing files in repo root: " + ", ".join(missing))
    st.stop()

# --------------------------------------------------------------------
# Helper: run subprocess and stream logs live to Streamlit
# --------------------------------------------------------------------
def stream_process(cmd, env=None):
    """Run a subprocess and yield stdout lines as they appear (text mode)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=env or os.environ.copy(),
    )

    for stdout_line in proc.stdout:
        yield stdout_line
    proc.stdout.close()
    returncode = proc.wait()
    yield f"\n[PROCESS FINISHED with return code {returncode}]\n"

# --------------------------------------------------------------------
# Main UI action
# --------------------------------------------------------------------
if st.button("🚀 RUN FULL SCANNER"):
    st.info("Starting scanner — output will stream below. This may take 1–4 minutes depending on network and compute.")
    log_area = st.empty()
    logs = ""

    # Ensure subprocess inherits Streamlit’s Python environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    # Run using same interpreter where all packages are installed
    cmd = [sys.executable, "run_all.py"]

    # Stream output in real time
    for chunk in stream_process(cmd, env):
        logs += chunk
        lines = logs.splitlines()
        # keep last 400 lines to keep UI responsive
        if len(lines) > 400:
            lines = lines[-400:]
            logs = "\n".join(lines) + "\n"
        log_area.code(logs)

    st.success("Run finished. If an Excel file was produced, it will appear below (if available).")

    # ----------------------------------------------------------------
    # Display download link for final Excel file if available
    # ----------------------------------------------------------------
    out_path = Path("Nifty200_Consolidated_Output.xlsx")
    if out_path.exists():
        with out_path.open("rb") as f:
            st.download_button(
                "📥 Download consolidated Excel",
                data=f,
                file_name=out_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.warning("Consolidated Excel not found. Check logs above for any errors.")
