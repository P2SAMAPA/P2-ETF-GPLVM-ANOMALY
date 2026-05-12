import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="GPLVM Manifold & Anomaly", layout="wide")
st.title("🌀 GPLVM + Warped GP Engine")
st.caption("Nonlinear latent manifold of ETFs + macro | Anomaly detection | Next‑day return prediction")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'gplvm_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No GPLVM results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** PCA + GP regression (GPLVM approximation)")

universes = data["universes"]

st.header("📈 Top 3 ETFs to Trade Tomorrow (by predicted return)")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.subheader(f"🌍 {universe_name}")
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.metric(f"#{idx+1} {etf['ticker']}", f"pred return = {etf['pred_return']:.4%}")
    # Show anomaly metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anomaly Score (MSE)", f"{uni_data['anomaly_mse']:.6f}")
    with col2:
        st.metric("Avg Predictive Variance", f"{uni_data['avg_predictive_variance']:.6f}")
    st.caption(f"Latent dimension used: {uni_data['latent_dim']}")
    st.divider()

st.caption("Higher predicted return → stronger buy signal. MSE = reconstruction error (higher = more anomalous).")
