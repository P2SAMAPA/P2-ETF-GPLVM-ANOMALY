"""
GPLVM approximation: PCA for latent manifold + GP regression for each observed dimension.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

class GPLVMEngine:
    def __init__(self, variance_threshold=0.95, kernel_length_scale=1.0, kernel_variance=1.0):
        self.variance_threshold = variance_threshold
        self.kernel = RBF(length_scale=kernel_length_scale) + WhiteKernel(noise_level=1e-3)
        self.scaler_X = StandardScaler()   # scale latent coordinates
        self.scaler_y = {}                 # per column scaler
        self.pca = None
        self.gp_models = {}                # column name -> GP regressor
        self.latent_dim = None
        self.feature_columns = None

    def fit(self, df):
        """
        df: DataFrame with shape (n_samples, n_features) – returns + macro
        """
        self.feature_columns = df.columns.tolist()
        X_raw = df.values   # n_samples x n_features

        # PCA to find latent coordinates
        self.pca = PCA(n_components=self.variance_threshold, svd_solver='full')
        Z = self.pca.fit_transform(X_raw)       # latent variable (n_samples x latent_dim)
        self.latent_dim = Z.shape[1]
        Z_scaled = self.scaler_X.fit_transform(Z)

        # Train one GP per original feature (warped GP)
        for i, col in enumerate(self.feature_columns):
            y = X_raw[:, i]
            # Scale output
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            self.scaler_y[col] = scaler_y
            # Train GP on Z_scaled -> y_scaled
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=2, random_state=42)
            gp.fit(Z_scaled, y_scaled)
            self.gp_models[col] = gp

        return self

    def reconstruct(self, df):
        """Return DataFrame of GP reconstructions for the same input."""
        Z = self.pca.transform(df.values)
        Z_scaled = self.scaler_X.transform(Z)
        pred_scaled = np.column_stack([self.gp_models[col].predict(Z_scaled) for col in self.feature_columns])
        # Inverse scaling
        pred = np.zeros_like(pred_scaled)
        for i, col in enumerate(self.feature_columns):
            pred[:, i] = self.scaler_y[col].inverse_transform(pred_scaled[:, i].reshape(-1, 1)).ravel()
        return pd.DataFrame(pred, index=df.index, columns=self.feature_columns)

    def get_anomaly_score(self, df):
        """Reconstruction MSE per day + average predictive variance."""
        Z = self.pca.transform(df.values)
        Z_scaled = self.scaler_X.transform(Z)
        mse_per_day = []
        var_per_day = []
        for i, col in enumerate(self.feature_columns):
            y_true = df[col].values
            y_pred, y_std = self.gp_models[col].predict(Z_scaled, return_std=True)
            y_pred = self.scaler_y[col].inverse_transform(y_pred.reshape(-1, 1)).ravel()
            mse = (y_true - y_pred) ** 2
            mse_per_day.append(mse)
            var_per_day.append(y_std ** 2)
        mse_total = np.mean(mse_per_day, axis=0)          # average across features per day
        avg_var = np.mean(var_per_day, axis=0)
        return mse_total, avg_var

    def predict_next_day_returns(self, df, last_n_days=20):
        """
        Predict next day's returns for all ETFs (exclude macro columns).
        Steps:
        1. Use last_n_days of latent coordinates to fit a GP over time (for each latent dim).
        2. Forecast next latent point.
        3. Reconstruct returns using GP models.
        """
        # Get latent trajectory
        Z = self.pca.transform(df.values)
        Z_scaled = self.scaler_X.transform(Z)
        # Use last_n_days
        Z_recent = Z_scaled[-last_n_days:, :]
        time_idx = np.arange(len(Z_recent)).reshape(-1, 1)

        # Predict next latent point (one day ahead)
        next_latent = np.zeros(self.latent_dim)
        for d in range(self.latent_dim):
            gp_time = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=2)
            gp_time.fit(time_idx, Z_recent[:, d])
            next_latent[d] = gp_time.predict(np.array([[len(Z_recent)]]), return_std=False)[0]

        # Reconstruct all features from next_latent
        next_latent_scaled = next_latent.reshape(1, -1)
        pred_scaled = np.column_stack([self.gp_models[col].predict(next_latent_scaled) for col in self.feature_columns])
        pred = np.zeros_like(pred_scaled)
        for i, col in enumerate(self.feature_columns):
            pred[0, i] = self.scaler_y[col].inverse_transform(pred_scaled[0, i].reshape(-1, 1)).ravel()[0]

        # Return only ETF returns (exclude macro columns for trading)
        etf_cols = [c for c in self.feature_columns if c in df.columns and c not in config.MACRO_COLUMNS]
        etf_returns = {col: pred[0, i] for i, col in enumerate(self.feature_columns) if col in etf_cols}
        return etf_returns
