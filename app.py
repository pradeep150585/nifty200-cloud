import streamlit as st
import subprocess
import time
import os

st.set_page_config(page_title="Nifty200 Scanner", layout="wide")

st.title("📊 NIFTY 200 Technical Scanner")
st.write("Run all timeframes (5M / 15M / 30M / Daily) in cloud and view console output.")

# -------- Function to run script and stream output -------- #
def run_script_live():
    process = subprocess.Popen(
        ["python", "run_all.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output_placeholder = st.empty()

    full_output = ""

    for line in process.stdout:
        full_output += line
        output_placeholder.text(full_output)

    errors = process.stderr.read()
    if errors.strip():
        st.error("⚠ Errors encountered:")
        st.text(errors)

    return full_output


# -------------------- UI -------------------- #
if st.button("🚀 RUN FULL SCANNER"):
    st.write("Running scripts... please wait. This may take a few minutes.")
    final_output = run_script_live()

    st.success("✅ SCAN COMPLETED")

    st.subheader("📥 Download Final Excel Output")
    if os.path.exists("Nifty200_Consolidated_Output.xlsx"):
        with open("Nifty200_Consolidated_Output.xlsx", "rb") as f:
            st.download_button(
                "Download Consolidated Excel",
                data=f,
                file_name="Nifty200_Consolidated_Output.xlsx"
            )
    else:
        st.warning("Final output file not generated.")