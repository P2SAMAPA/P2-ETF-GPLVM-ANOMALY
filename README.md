# GPLVM + Warped GP Engine

**Nonlinear latent manifold learning for ETFs + macro factors.**  
Predicts next‑day returns, detects anomalies via reconstruction error.

- **Auto‑selects latent dimension** (PCA explained variance ≥95%)
- **Gaussian Process regression** for each original column (warping)
- **Rolling 252‑day window**, refit daily
- Outputs top 3 ETFs per universe (FI, Equity, Combined)
- Streamlit dashboard with anomaly scores and latent dimension info

## Run locally
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python trainer.py
streamlit run streamlit_app.py
