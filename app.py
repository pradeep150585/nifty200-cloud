# app.py
import streamlit as st
import subprocess
import os
import time
from pathlib import Path

st.set_page_config(page_title="Nifty200 Scanner", layout="wide")
st.title("📊 NIFTY 200 Technical Scanner (Streamlit Community Cloud)")
st.write("Click **Run Full Scanner** to run your pipeline. Output is streamed below.")

# Safety: required files check
required = ["run_all.py", "5M.py", "15M.py", "30M.py", "Daily.py"]
missing = [f for f in required if not Path(f).exists()]
if missing:
    st.error("Missing files in repo root: " + ", ".join(missing))
    st.stop()

def stream_process(cmd):
    """Run subprocess and yield stdout lines as they appear (text mode)."""
    # Use universal_newlines/text mode to get strings, and bufsize=1 for line buffering
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            bufsize=1,
                            universal_newlines=True,
                            env=os.environ.copy())

    # Stream every line
    for stdout_line in proc.stdout:
        yield stdout_line
    proc.stdout.close()
    returncode = proc.wait()
    yield f"\n[PROCESS FINISHED with return code {returncode}]\n"

if st.button("🚀 RUN FULL SCANNER"):
    st.info("Starting scanner — output will stream below. This may take 1–4 minutes depending on network and compute.")
    log_area = st.empty()
    logs = ""
    # Run via same Python executable
    py = os.getenv("PYTHON", "python")
    cmd = [py, "run_all.py"]
    for chunk in stream_process(cmd):
        logs += chunk
        # Keep only last ~200 lines in UI to keep it responsive
        lines = logs.splitlines()
        if len(lines) > 400:
            lines = lines[-400:]
            logs = "\n".join(lines) + "\n"
        # show as code block for fixed-width
        log_area.code(logs)
    st.success("Run finished. If an Excel file was produced, it will appear below (if available).")

    out_path = Path("Nifty200_Consolidated_Output.xlsx")
    if out_path.exists():
        with out_path.open("rb") as f:
            st.download_button("Download consolidated Excel", data=f, file_name=out_path.name)
    else:
        st.warning("Consolidated Excel not found. Check logs for errors.")
