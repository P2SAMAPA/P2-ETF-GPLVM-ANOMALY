"""
Streamlit dashboard – GPLVM + Warped GP Engine
Professional layout with cards, metrics, and expandable details.
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

# Page configuration
st.set_page_config(
    page_title="GPLVM Manifold Engine",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .universe-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .etf-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .etf-ticker {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .etf-return {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .positive {
        color: #00cc96;
    }
    .negative {
        color: #ef553b;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌀 GPLVM + Warped GP Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Nonlinear latent manifold learning | Anomaly detection | Next‑day return prediction</div>', unsafe_allow_html=True)

# Sidebar – clean and compact, no large logo
st.sidebar.markdown("## 🧠 GPLVM Engine")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** PCA + GP regression")
st.sidebar.markdown("**Latent dim:** auto‑selected (≥95% variance)")
st.sidebar.markdown("---")
st.sidebar.caption("Data: [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)")

# Load data
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
    st.error("No GPLVM results found. Run `trainer.py` first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

# Store run date in session state for sidebar
st.session_state['run_date'] = data['run_date']

universes = data["universes"]
if not universes:
    st.warning("No universe data available.")
    st.stop()

# Main content
st.header("📈 Top ETFs to Trade Tomorrow")
st.markdown("*Ranked by predicted next‑day return from latent manifold extrapolation.*")

# Display each universe as a separate card deck
for universe_name, uni_data in universes.items():
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.info(f"No predictions for {universe_name}")
        continue
    
    # Create 3 columns for top 3 ETFs
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            ticker = etf["ticker"]
            pred_return = etf["pred_return"]
            # Color based on return
            color_class = "positive" if pred_return > 0 else "negative"
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{ticker}</div>
                <div class="etf-return">Predicted return<br><span class="{color_class}">{pred_return:.2%}</span></div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show anomaly metrics in a row of two columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Anomaly Score (Reconstruction MSE)", f"{uni_data['anomaly_mse']:.6f}")
    with col2:
        st.metric("Average Predictive Variance", f"{uni_data['avg_predictive_variance']:.6f}")
    st.caption(f"Latent dimension used: {uni_data['latent_dim']}")
    st.markdown("---")

# Historical anomaly plot (if multiple JSON files exist)
st.header("📉 Anomaly Score History (Last 30 days)")
with st.spinner("Loading historical data..."):
    # Get all JSON files, sort by date, take last 30
    json_files = [f for f in files if f.endswith('.json') and 'gplvm_' in f]
    json_files.sort(reverse=True)
    history_data = []
    for fname in json_files[:30]:
        try:
            fs = HfFileSystem(token=HF_TOKEN)
            with fs.open(f"datasets/{OUTPUT_REPO}/{fname}", "r") as f:
                hist = json.load(f)
                run_date = hist['run_date']
                for uni, val in hist['universes'].items():
                    history_data.append({
                        "date": run_date,
                        "universe": uni,
                        "anomaly_mse": val['anomaly_mse'],
                        "avg_var": val['avg_predictive_variance']
                    })
        except:
            pass
    if history_data:
        df_hist = pd.DataFrame(history_data)
        fig = px.line(df_hist, x="date", y="anomaly_mse", color="universe", 
                      title="Reconstruction MSE over time",
                      labels={"anomaly_mse": "MSE (lower=more normal)", "date": "Run Date"})
        fig.update_layout(height=400, legend_title="Universe")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to plot chart.")

st.caption("Higher predicted return → stronger buy signal. MSE = reconstruction error (higher = more anomalous regime).")
