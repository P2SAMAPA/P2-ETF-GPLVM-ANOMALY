"""
Daily training: rolling 252-day fit, predict next-day ETF returns, output top 3 per universe.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from gplvm_engine import GPLVMEngine

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        universe_df = data_manager.prepare_universe_data(df, tickers)
        if universe_df.empty or len(universe_df) < config.ROLLING_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Use last ROLLING_WINDOW days for training
        train_df = universe_df.iloc[-config.ROLLING_WINDOW:]

        # Fit GPLVM engine
        engine = GPLVMEngine(
            variance_threshold=config.LATENT_VARIANCE_THRESHOLD,
            kernel_length_scale=1.0,
            kernel_variance=1.0
        )
        engine.fit(train_df)

        # Anomaly scores for the last day of training (or we can compute on whole train)
        recon_mse, avg_var = engine.get_anomaly_score(train_df)
        latest_mse = float(recon_mse[-1])
        latest_var = float(avg_var[-1])

        # Predict next day returns for ETFs (exclude macro)
        predicted_returns = engine.predict_next_day_returns(train_df, last_n_days=config.N_LATENT_FORECAST_DAYS)

        # Sort ETFs by predicted return, take top 3
        sorted_etfs = sorted(predicted_returns.items(), key=lambda x: x[1], reverse=True)
        top3 = [{"ticker": ticker, "pred_return": round(ret, 6)} for ticker, ret in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs for {universe_name}:")
        for idx, etf in enumerate(top3, 1):
            print(f"    {idx}. {etf['ticker']} (pred_return={etf['pred_return']})")
        print(f"  Anomaly score (reconstruction MSE): {latest_mse:.6f}, Avg var: {latest_var:.6f}")

        all_results[universe_name] = {
            "top_etfs": top3,
            "anomaly_mse": latest_mse,
            "avg_predictive_variance": latest_var,
            "latent_dim": engine.latent_dim,
            "run_date": today
        }

    # Save to JSON
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/gplvm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== GPLVM Engine complete ===")

if __name__ == "__main__":
    main()
