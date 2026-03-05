"""
streamlit_app.py
================
A beautiful, interactive Streamlit web application for training and evaluating
ANN models on chemical engineering datasets (DWSIM-compatible output).

Features:
  - Upload CSV / Excel dataset
  - Interactive column selection for inputs & outputs
  - Hyperparameter configuration panel
  - Live epoch-by-epoch training progress bar & loss chart (auto-refresh)
  - Comprehensive final report:
        * Training / Validation loss curves
        * RMSE, MAE, R² per output
        * Parity plots (Actual vs Predicted) for every output
        * Residual plots
        * Feature importance (permutation-based, fast)
  - Export model package (model.dat + scalers.dat + config.json) as ZIP

Run:
    streamlit run streamlit_app.py
"""

import io
import json
import struct
import time
import zipfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DWSIM ANN Model Studio",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(8px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.55);
        margin-top: 4px;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(167,139,250,0.15) 0%, rgba(96,165,250,0.15) 100%);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 28px;
        text-align: center;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        color: rgba(255,255,255,0.55);
        font-size: 1.05rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #a78bfa;
        border-left: 3px solid #a78bfa;
        padding-left: 12px;
        margin: 24px 0 14px 0;
    }

    /* Report box */
    .report-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 20px;
    }

    /* Tag badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px;
    }
    .badge-in  { background: rgba(96,165,250,0.2);  color: #60a5fa; border: 1px solid #60a5fa44; }
    .badge-out { background: rgba(52,211,153,0.2);  color: #34d399; border: 1px solid #34d39944; }

    /* Streamlit widgets de-jank */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s, transform 0.15s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 14px 18px;
    }

    /* Plotly / matplotlib container */
    .plot-container {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

DARK_FIG_STYLE = {
    "facecolor": "#1a1a2e",
    "edgecolor": "none",
}
DARK_AX_STYLE = {
    "facecolor": "#16213e",
    "titlecolor": "white",
}
GRID_COLOR = "#2d2d4e"
TEXT_COLOR = "white"
ACCENT1 = "#a78bfa"
ACCENT2 = "#60a5fa"
ACCENT3 = "#34d399"
ACCENT4 = "#f97316"


def styled_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=DARK_FIG_STYLE["facecolor"], **kwargs)
    return fig


def style_ax(ax):
    ax.set_facecolor(DARK_AX_STYLE["facecolor"])
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes, activation="tanh", dropout=0.0, use_bn=False):
        super().__init__()
        layers = []
        last = n_in
        ACT_MAP = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "relu": nn.ReLU,
            "leakyrelu": lambda: nn.LeakyReLU(0.01),
            "elu": nn.ELU,
            "selu": nn.SELU,
            "gelu": nn.GELU,
        }
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACT_MAP.get(activation, nn.Tanh)())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "leakyrelu", "elu", "selu", "gelu"):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training function  (yields per-epoch results for live updates)
# ─────────────────────────────────────────────────────────────────────────────

def train_model_generator(
    df, input_cols, output_cols, hidden_layers, epochs, lr,
    activation, dropout, use_bn, batch_size, val_split=0.15
):
    """Trains the model and yields a dict at every epoch for live monitoring."""
    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    # ── Drop rows that contain NaN or Inf in either X or y ────────────────────
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped > 0:
        # Signal the caller so a warning can be shown in the UI
        yield {"warning": f"⚠️ {n_dropped} row(s) with NaN / Inf values were removed before training."}
    X = X[finite_mask]
    y = y[finite_mask]

    if len(X) < 10:
        raise ValueError(
            f"After removing invalid rows only {len(X)} samples remain — "
            "please check your dataset for missing or infinite values."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    Xtn = scaler_x.transform(X_train)
    Xvn = scaler_x.transform(X_val)
    ytn = scaler_y.transform(y_train)
    yvn = scaler_y.transform(y_val)

    Xt = torch.from_numpy(Xtn)
    yt = torch.from_numpy(ytn)
    Xv = torch.from_numpy(Xvn)
    yv = torch.from_numpy(yvn)

    model = SimpleMLP(X.shape[1], y.shape[1], hidden_layers, activation, dropout, use_bn)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, epochs // 20)
    )

    train_losses, val_losses = [], []
    best_model_state = None
    best_val_loss = float("inf")

    dataset = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(np.mean(ep_losses))

        model.eval()
        with torch.no_grad():
            vp = model(Xv).numpy()
            val_loss = float(loss_fn(torch.from_numpy(vp), yv).item())

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        # De-normalise val predictions for RMSE
        vp_orig = scaler_y.inverse_transform(vp)
        val_rmse = float(np.sqrt(np.mean((vp_orig - y_val) ** 2)))

        yield {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "train_losses": list(train_losses),
            "val_losses": list(val_losses),
        }

    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_val_pred_n = model(Xv).numpy()
        y_train_pred_n = model(Xt).numpy()

    y_val_pred = scaler_y.inverse_transform(y_val_pred_n)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_n)

    # Full-dataset predictions for report
    Xall = torch.from_numpy(scaler_x.transform(X))
    with torch.no_grad():
        y_all_pred_n = model(Xall).numpy()
    y_all_pred = scaler_y.inverse_transform(y_all_pred_n)

    final = {
        "done": True,
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_all": y,
        "y_all_pred": y_all_pred,
    }
    yield final


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

def export_model_binary(model):
    buf = io.BytesIO()
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    buf.write(struct.pack("i", len(linear_layers)))
    for layer in linear_layers:
        w = layer.weight.data.cpu().numpy()
        b = layer.bias.data.cpu().numpy()
        buf.write(struct.pack("i", w.shape[0]))
        buf.write(struct.pack("i", w.shape[1]))
        for v in w.flatten():
            buf.write(struct.pack("d", float(v)))
        for v in b:
            buf.write(struct.pack("d", float(v)))
    return buf.getvalue()


def export_scalers_binary(scaler_x, scaler_y):
    buf = io.BytesIO()
    for sc in [scaler_x, scaler_y]:
        m, s = sc.mean_, sc.scale_
        buf.write(struct.pack("i", len(m)))
        for v in m:
            buf.write(struct.pack("d", float(v)))
        for v in s:
            buf.write(struct.pack("d", float(v)))
    return buf.getvalue()


def build_zip(model, scaler_x, scaler_y, input_cols, output_cols, activation, hidden_layers):
    zio = io.BytesIO()
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    config = {
        "name": "DWSIM ANN Model (Streamlit Export)",
        "description": "External neural network model for DWSIM integration",
        "input_count": len(input_cols),
        "output_count": len(output_cols),
        "activation": activation,
        "model_file": "model.dat",
        "scaler_file": "scalers.dat",
        "hidden_layers": hidden_layers,
        "input_names": input_cols,
        "output_names": output_cols,
    }
    with zipfile.ZipFile(zio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.dat", export_model_binary(model))
        zf.writestr("scalers.dat", export_scalers_binary(scaler_x, scaler_y))
        zf.writestr("config.json", json.dumps(config, indent=2))
    zio.seek(0)
    return zio.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses):
    fig = styled_fig(figsize=(8, 3.5))
    ax = fig.add_subplot(111)
    style_ax(ax)
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, color=ACCENT1, linewidth=2, label="Train Loss")
    ax.plot(epochs, val_losses, color=ACCENT2, linewidth=2, label="Val Loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss Curve", pad=12)
    ax.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_parity(y_true, y_pred, col_name):
    fig = styled_fig(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    style_ax(ax)
    ax.scatter(y_true, y_pred, alpha=0.65, s=22, color=ACCENT1, edgecolors="none")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Parity — {col_name}", pad=10)
    ax.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, col_name):
    residuals = y_pred - y_true
    fig = styled_fig(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111)
    style_ax(ax)
    ax.scatter(y_pred, residuals, alpha=0.65, s=22, color=ACCENT3, edgecolors="none")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Pred − Actual)")
    ax.set_title(f"Residuals — {col_name}", pad=10)
    fig.tight_layout()
    return fig


def quick_loss_chart(train_losses, val_losses):
    """Tiny inline chart used during live training."""
    fig = styled_fig(figsize=(7, 2.5))
    ax = fig.add_subplot(111)
    style_ax(ax)
    e = list(range(1, len(train_losses) + 1))
    ax.plot(e, train_losses, color=ACCENT1, linewidth=1.5, label="Train")
    ax.plot(e, val_losses, color=ACCENT2, linewidth=1.5, linestyle="--", label="Val")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Loss", fontsize=9)
    ax.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig


def compute_metrics(y_true, y_pred, output_cols):
    rows = []
    for i, col in enumerate(output_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        # Guard against NaN in predictions before passing to sklearn
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt_c, yp_c = yt[mask], yp[mask]
        if len(yt_c) < 2:
            rows.append({"Output": col, "RMSE": float("nan"), "MAE": float("nan"),
                         "R²": float("nan"), "MAPE (%)": float("nan")})
            continue
        rmse = float(np.sqrt(mean_squared_error(yt_c, yp_c)))
        mae = float(mean_absolute_error(yt_c, yp_c))
        r2 = float(r2_score(yt_c, yp_c))
        mape = float(np.mean(np.abs((yt_c - yp_c) / (np.abs(yt_c) + 1e-8))) * 100)
        rows.append({"Output": col, "RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mape})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Additional Chart Helpers — Actual vs Predicted
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_pred_line(y_true, y_pred, col_name, split_label="Validation"):
    """
    Line / scatter chart with sample index on X-axis.
    Shows actual vs predicted as overlapping lines — mirrors the Tkinter GUI style.
    """
    n = len(y_true)
    idx = np.arange(n)
    fig = styled_fig(figsize=(9, 3.5))
    ax = fig.add_subplot(111)
    style_ax(ax)
    ax.plot(idx, y_true, color=ACCENT3, linewidth=1.6, label="Actual", alpha=0.9)
    ax.plot(idx, y_pred, color=ACCENT4, linewidth=1.6, label="Predicted",
            linestyle="--", alpha=0.9)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel(col_name)
    ax.set_title(f"{col_name} — Actual vs Predicted ({split_label})", pad=10)
    ax.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_parity_residuals_panel(y_true, y_pred, col_name):
    """
    Side-by-side panel: parity scatter (left) + residual scatter (right).
    """
    fig = styled_fig(figsize=(10, 4))
    axes = fig.subplots(1, 2)

    # ── Left: parity ───────────────────────────────────────────────────────────────
    ax_p = axes[0]
    style_ax(ax_p)
    ax_p.scatter(y_true, y_pred, alpha=0.65, s=20, color=ACCENT1, edgecolors="none")
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax_p.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    ax_p.set_xlabel("Actual")
    ax_p.set_ylabel("Predicted")
    ax_p.set_title(f"Parity — {col_name}", pad=8)
    ax_p.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # ── Right: residuals ────────────────────────────────────────────────────────
    ax_r = axes[1]
    style_ax(ax_r)
    residuals = y_pred - y_true
    ax_r.scatter(y_pred, residuals, alpha=0.65, s=20, color=ACCENT3, edgecolors="none")
    ax_r.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax_r.set_xlabel("Predicted")
    ax_r.set_ylabel("Residual (Pred − Actual)")
    ax_r.set_title(f"Residuals — {col_name}", pad=8)

    fig.tight_layout(pad=2.0)
    return fig


def plot_all_outputs_overview(y_true, y_pred, output_cols):
    """
    One subplot per output: line chart of Actual vs Predicted (sample index).
    Compact multi-panel view for quick inspection after training.
    """
    n_out = len(output_cols)
    ncols = min(n_out, 2)
    nrows = (n_out + ncols - 1) // ncols
    fig = styled_fig(figsize=(10 * ncols / 2, 3.5 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)

    idx = np.arange(len(y_true))
    for i, col in enumerate(output_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        style_ax(ax)
        ax.plot(idx, y_true[:, i], color=ACCENT3, linewidth=1.4,
                label="Actual", alpha=0.9)
        ax.plot(idx, y_pred[:, i], color=ACCENT4, linewidth=1.4,
                linestyle="--", label="Predicted", alpha=0.9)
        ax.set_title(col, pad=6)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.legend(facecolor="#1a1a2e", edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=7)

    # Hide unused subplots
    for j in range(n_out, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Actual vs Predicted — All Outputs (Validation Set)",
                 color=TEXT_COLOR, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Permutation-based feature importance (fast, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(model, scaler_x, scaler_y, X, y, input_cols, n_repeats=5):
    model.eval()
    Xn = scaler_x.transform(X)
    Xt = torch.from_numpy(Xn.astype(np.float32))
    with torch.no_grad():
        base_pred = scaler_y.inverse_transform(model(Xt).numpy())
    base_rmse = float(np.sqrt(np.mean((base_pred - y) ** 2)))

    importances = []
    for fi in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            Xp = Xn.copy()
            np.random.shuffle(Xp[:, fi])
            Xpt = torch.from_numpy(Xp.astype(np.float32))
            with torch.no_grad():
                pp = scaler_y.inverse_transform(model(Xpt).numpy())
            scores.append(float(np.sqrt(np.mean((pp - y) ** 2))) - base_rmse)
        importances.append(float(np.mean(scores)))
    return importances


def plot_feature_importance(importances, input_cols):
    idx = np.argsort(importances)
    fig = styled_fig(figsize=(7, max(3.5, len(input_cols) * 0.4 + 1)))
    ax = fig.add_subplot(111)
    style_ax(ax)
    colors = [ACCENT1 if v >= 0 else ACCENT4 for v in [importances[i] for i in idx]]
    ax.barh([input_cols[i] for i in idx], [importances[i] for i in idx], color=colors, edgecolor="none")
    ax.set_xlabel("Mean ΔRMSE (higher = more important)")
    ax.set_title("Permutation Feature Importance", pad=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "df": None,
        "filename": "",
        "input_cols": [],
        "output_cols": [],
        "training_done": False,
        "training_results": None,
        "hyperparams": {
            "lr": 1e-3,
            "epochs": 200,
            "hidden": "64,32",
            "activation": "tanh",
            "dropout": 0.0,
            "use_bn": False,
            "batch_size": 32,
            "val_split": 0.15,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">⚗️ DWSIM ANN Model Studio</div>
        <div class="hero-sub">
            Upload · Configure · Train · Evaluate · Export — All in one beautiful interface
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — dataset & column selection
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📁 Dataset")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload your chemical engineering dataset.",
    )

    if uploaded:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)

            if st.session_state.filename != uploaded.name:
                st.session_state.df = df_new
                st.session_state.filename = uploaded.name
                st.session_state.input_cols = []
                st.session_state.output_cols = []
                st.session_state.training_done = False
                st.session_state.training_results = None

            df = st.session_state.df
            st.success(f"✅ Loaded **{uploaded.name}**")
            st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df = None
    else:
        df = st.session_state.df

    if df is not None:
        all_cols = list(df.columns)
        st.markdown("### 🔢 Select Input Columns (X)")
        input_cols = st.multiselect(
            "Input features",
            all_cols,
            default=st.session_state.input_cols or [],
            key="input_sel",
        )
        remaining = [c for c in all_cols if c not in input_cols]

        st.markdown("### 🎯 Select Output Columns (Y)")
        output_cols = st.multiselect(
            "Target outputs",
            remaining,
            default=[c for c in st.session_state.output_cols if c in remaining],
            key="output_sel",
        )
        st.session_state.input_cols = input_cols
        st.session_state.output_cols = output_cols

        if input_cols:
            st.markdown(
                "**Inputs:** " + " ".join(f'<span class="badge badge-in">{c}</span>' for c in input_cols),
                unsafe_allow_html=True,
            )
        if output_cols:
            st.markdown(
                "**Outputs:** " + " ".join(f'<span class="badge badge-out">{c}</span>' for c in output_cols),
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## ⚙️ Hyperparameters")

    hp = st.session_state.hyperparams
    hp["lr"] = st.number_input("Learning rate", value=hp["lr"], format="%.5f", min_value=1e-6, max_value=1.0, step=1e-4)
    hp["epochs"] = st.slider("Epochs", 10, 2000, int(hp["epochs"]), step=10)
    hp["hidden"] = st.text_input("Hidden layers (comma-separated)", value=hp["hidden"], help="e.g.  64,32  or  128,64,32")
    hp["activation"] = st.selectbox("Activation", ["tanh", "relu", "leakyrelu", "elu", "selu", "gelu", "sigmoid"], index=["tanh", "relu", "leakyrelu", "elu", "selu", "gelu", "sigmoid"].index(hp["activation"]))
    hp["dropout"] = st.slider("Dropout", 0.0, 0.5, float(hp["dropout"]), step=0.01)
    hp["use_bn"] = st.checkbox("Batch Normalisation", value=bool(hp["use_bn"]))
    hp["batch_size"] = st.select_slider("Batch size", options=[8, 16, 32, 64, 128, 256], value=int(hp["batch_size"]))
    hp["val_split"] = st.slider("Validation split", 0.05, 0.40, float(hp["val_split"]), step=0.01)
    st.session_state.hyperparams = hp

# ─────────────────────────────────────────────────────────────────────────────
# MAIN — tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_data, tab_train, tab_report, tab_predict = st.tabs(
    ["📊 Data Preview", "🚀 Train Model", "📋 Model Report", "🔮 Predict"]
)

# ── Tab 1: Data Preview ────────────────────────────────────────────────────────
with tab_data:
    if df is None:
        st.info("👈 Upload a dataset from the sidebar to get started.")
    else:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Selected Inputs", len(st.session_state.input_cols))
        c4.metric("Selected Outputs", len(st.session_state.output_cols))

        st.dataframe(df.head(50), use_container_width=True, height=320)

        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T.style.background_gradient(cmap="Blues"), use_container_width=True)

        # Correlation heatmap (only numeric)
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
            corr = num_df.corr()
            fig = styled_fig(figsize=(max(6, corr.shape[0] * 0.55), max(5, corr.shape[0] * 0.5)))
            ax = fig.add_subplot(111)
            style_ax(ax)
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", color=TEXT_COLOR, fontsize=8)
            ax.set_yticklabels(corr.columns, color=TEXT_COLOR, fontsize=8)
            fig.colorbar(im, ax=ax)
            ax.set_title("Feature Correlation Matrix", color=TEXT_COLOR, pad=12)
            fig.tight_layout()
            st.pyplot(fig)

# ── Tab 2: Train Model ─────────────────────────────────────────────────────────
with tab_train:
    if df is None:
        st.info("👈 Upload a dataset to begin.")
    elif not st.session_state.input_cols or not st.session_state.output_cols:
        st.warning("⚠️ Please select at least one input and one output column in the sidebar.")
    else:
        # Parse hidden layers
        try:
            hidden_layers = [int(x.strip()) for x in hp["hidden"].split(",") if x.strip()]
            if not hidden_layers:
                raise ValueError
        except Exception:
            st.error("❌ Hidden layers must be comma-separated integers (e.g. 64,32).")
            st.stop()

        arch_str = " → ".join(
            [f"In({len(st.session_state.input_cols)})"]
            + [f"H{i+1}({h})" for i, h in enumerate(hidden_layers)]
            + [f"Out({len(st.session_state.output_cols)})"]
        )
        st.markdown(f'<div class="section-header">Architecture: {arch_str}</div>', unsafe_allow_html=True)

        train_btn = st.button("🚀 Start Training", use_container_width=True)

        if train_btn:
            st.session_state.training_done = False
            st.session_state.training_results = None

            # Live training UI
            prog_bar = st.progress(0, text="Starting…")
            status_row = st.columns(4)
            ep_txt      = status_row[0].empty()
            tloss_txt   = status_row[1].empty()
            vloss_txt   = status_row[2].empty()
            rmse_txt    = status_row[3].empty()
            chart_ph    = st.empty()

            results = None
            gen = train_model_generator(
                df=df,
                input_cols=st.session_state.input_cols,
                output_cols=st.session_state.output_cols,
                hidden_layers=hidden_layers,
                epochs=hp["epochs"],
                lr=hp["lr"],
                activation=hp["activation"],
                dropout=hp["dropout"],
                use_bn=hp["use_bn"],
                batch_size=hp["batch_size"],
                val_split=hp["val_split"],
            )

            REFRESH_EVERY = max(1, hp["epochs"] // 100)  # update UI ~100 times max

            for step in gen:
                if step.get("done"):
                    results = step
                    break
                # Dataset cleaning warning (emitted before epoch 1)
                if step.get("warning"):
                    st.warning(step["warning"])
                    continue
                ep = step["epoch"]
                total = hp["epochs"]
                prog_bar.progress(ep / total, text=f"Epoch {ep}/{total}")
                ep_txt.metric("Epoch", f"{ep}/{total}")
                tloss_txt.metric("Train Loss", f"{step['train_loss']:.5f}")
                vloss_txt.metric("Val Loss", f"{step['val_loss']:.5f}")
                rmse_txt.metric("Val RMSE", f"{step['val_rmse']:.4f}")

                if ep % REFRESH_EVERY == 0 or ep == total:
                    fig = quick_loss_chart(step["train_losses"], step["val_losses"])
                    chart_ph.pyplot(fig)
                    plt.close(fig)

            prog_bar.progress(1.0, text="✅ Training complete!")
            st.session_state.training_done = True
            st.session_state.training_results = results
            st.success("🎉 Training finished! Navigate to **📋 Model Report** for the full analysis.")

        if st.session_state.training_done and st.session_state.training_results:
            res = st.session_state.training_results
            out_cols = st.session_state.output_cols

            st.markdown("### 📊 Quick Metrics (Validation Set)")
            mdf = compute_metrics(res["y_val"], res["y_val_pred"], out_cols)
            st.dataframe(mdf.set_index("Output").style.format("{:.4f}"), use_container_width=True)

            # ───────────────────────────────────────────────────────────────────
            # Actual vs Predicted charts — Train tab
            # ───────────────────────────────────────────────────────────────────
            st.markdown(
                '<div class="section-header">📈 Actual vs Predicted — Validation Set</div>',
                unsafe_allow_html=True,
            )

            # ── 1. Overview: all outputs in one figure ───────────────────────────────
            fig_ov = plot_all_outputs_overview(res["y_val"], res["y_val_pred"], out_cols)
            st.pyplot(fig_ov, use_container_width=True)
            plt.close(fig_ov)

            # ── 2. Per-output: line chart + parity + residuals ──────────────────────
            selected_output = st.selectbox(
                "🔍 Select output to inspect in detail",
                out_cols,
                key="train_output_select",
            )
            oi = out_cols.index(selected_output)

            # Line chart — actual vs predicted (sample index)
            fig_line = plot_actual_vs_pred_line(
                res["y_val"][:, oi], res["y_val_pred"][:, oi],
                selected_output, split_label="Validation"
            )
            st.pyplot(fig_line, use_container_width=True)
            plt.close(fig_line)

            # Parity + residuals side by side
            fig_pr = plot_parity_residuals_panel(
                res["y_val"][:, oi], res["y_val_pred"][:, oi],
                selected_output
            )
            st.pyplot(fig_pr, use_container_width=True)
            plt.close(fig_pr)

            # ── 3. Train set comparison ────────────────────────────────────────────
            with st.expander("Show Training-set Actual vs Predicted"):
                fig_tr = plot_actual_vs_pred_line(
                    res["y_train"][:, oi], res["y_train_pred"][:, oi],
                    selected_output, split_label="Training"
                )
                st.pyplot(fig_tr, use_container_width=True)
                plt.close(fig_tr)

# ── Tab 3: Model Report ────────────────────────────────────────────────────────
with tab_report:
    if not st.session_state.training_done or st.session_state.training_results is None:
        st.info("⚠️ Train the model first (go to **🚀 Train Model** tab).")
    else:
        res = st.session_state.training_results
        out_cols = st.session_state.output_cols
        in_cols  = st.session_state.input_cols

        # ── Summary header ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Comprehensive Model Report</div>', unsafe_allow_html=True)

        # Dataset size breakdown
        total_n = len(res["y_all"])
        val_n   = len(res["y_val"])
        train_n = total_n - val_n

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples", f"{total_n:,}")
        c2.metric("Train Samples", f"{train_n:,}")
        c3.metric("Val Samples",   f"{val_n:,}")
        c4.metric("Architecture",  hp["hidden"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Activation",    hp["activation"].upper())
        c6.metric("Learning Rate", f"{hp['lr']:.5f}")
        c7.metric("Epochs Run",    len(res["train_losses"]))
        c8.metric("Batch Norm",    "Yes" if hp["use_bn"] else "No")

        # ── Loss curves ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📉 Loss Curves</div>', unsafe_allow_html=True)
        fig_loss = plot_loss_curves(res["train_losses"], res["val_losses"])
        st.pyplot(fig_loss, use_container_width=True)
        plt.close(fig_loss)

        # ── Per-output metrics ───────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Per-Output Metrics</div>', unsafe_allow_html=True)

        mdf_val   = compute_metrics(res["y_val"],   res["y_val_pred"],   out_cols)
        mdf_train = compute_metrics(res["y_train"], res["y_train_pred"], out_cols)

        col_v, col_t = st.columns(2)
        col_v.markdown("**Validation Set**")
        col_v.dataframe(
            mdf_val.set_index("Output").style.format("{:.4f}").background_gradient(cmap="Purples", subset=["R²"]),
            use_container_width=True,
        )
        col_t.markdown("**Training Set**")
        col_t.dataframe(
            mdf_train.set_index("Output").style.format("{:.4f}").background_gradient(cmap="Blues", subset=["R²"]),
            use_container_width=True,
        )

        # Overall summary card
        avg_r2   = mdf_val["R²"].mean()
        avg_rmse = mdf_val["RMSE"].mean()
        avg_mae  = mdf_val["MAE"].mean()
        train_final_loss = res["train_losses"][-1]
        val_final_loss   = res["val_losses"][-1]

        st.markdown(
            f"""
            <div class="report-box">
            <b style="color:#a78bfa">Overall Summary (Validation)</b><br><br>
            <table style="width:100%; color:white; font-size:0.95rem;">
            <tr>
              <td>Avg R²</td><td><b style="color:#34d399">{avg_r2:.4f}</b></td>
              <td>Avg RMSE</td><td><b style="color:#60a5fa">{avg_rmse:.4f}</b></td>
              <td>Avg MAE</td><td><b style="color:#f97316">{avg_mae:.4f}</b></td>
            </tr>
            <tr>
              <td>Final Train Loss</td><td><b>{train_final_loss:.6f}</b></td>
              <td>Final Val Loss</td><td><b>{val_final_loss:.6f}</b></td>
              <td>Inputs / Outputs</td><td><b>{len(in_cols)} / {len(out_cols)}</b></td>
            </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Parity + Residual plots ──────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Parity & Residual Plots (per output)</div>', unsafe_allow_html=True)
        for col in out_cols:
            idx = out_cols.index(col)
            yt = res["y_val"][:, idx]
            yp = res["y_val_pred"][:, idx]
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(plot_parity(yt, yp, col), use_container_width=True)
            with c2:
                st.pyplot(plot_residuals(yt, yp, col), use_container_width=True)

        # ── Feature Importance ───────────────────────────────────────────────
        st.markdown('<div class="section-header">🔍 Feature Importance (Permutation)</div>', unsafe_allow_html=True)
        with st.spinner("Computing feature importance…"):
            X_all = df[in_cols].values.astype(np.float32)
            y_all = df[out_cols].values.astype(np.float32)
            importances = permutation_importance(
                res["model"], res["scaler_x"], res["scaler_y"], X_all, y_all, in_cols
            )
        fig_imp = plot_feature_importance(importances, in_cols)
        st.pyplot(fig_imp, use_container_width=True)
        plt.close(fig_imp)

        # ── Error distribution histogram ─────────────────────────────────────
        st.markdown('<div class="section-header">📐 Error Distribution (All Outputs)</div>', unsafe_allow_html=True)
        residuals_all = (res["y_val_pred"] - res["y_val"]).flatten()

        # Guard: remove non-finite residuals before plotting
        finite_res = residuals_all[np.isfinite(residuals_all)]
        n_bad_res = len(residuals_all) - len(finite_res)
        if n_bad_res > 0:
            st.warning(
                f"⚠️ {n_bad_res} non-finite residual value(s) (NaN/Inf) were excluded from the histogram. "
                "This usually means some predictions diverged — consider reducing the learning rate, "
                "adding Batch Normalisation, or cleaning your dataset."
            )

        if len(finite_res) == 0:
            st.error("All residuals are non-finite — cannot plot histogram. Please retrain with a cleaned dataset.")
        else:
            fig_hist = styled_fig(figsize=(8, 3.2))
            ax_h = fig_hist.add_subplot(111)
            style_ax(ax_h)
            ax_h.hist(finite_res, bins=40, color=ACCENT1, alpha=0.8, edgecolor="none")
            ax_h.axvline(0, color="red", linewidth=1.5, linestyle="--")
            ax_h.set_xlabel("Residual (Predicted − Actual)")
            ax_h.set_ylabel("Count")
            ax_h.set_title("Residual Distribution Across All Outputs")
            fig_hist.tight_layout()
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)

        # ── Export ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📦 Export Model Package</div>', unsafe_allow_html=True)
        try:
            hidden_layers = [int(x.strip()) for x in hp["hidden"].split(",") if x.strip()]
        except Exception:
            hidden_layers = [64, 32]

        zip_bytes = build_zip(
            model=res["model"],
            scaler_x=res["scaler_x"],
            scaler_y=res["scaler_y"],
            input_cols=in_cols,
            output_cols=out_cols,
            activation=hp["activation"],
            hidden_layers=hidden_layers,
        )
        st.download_button(
            label="⬇️ Download Model ZIP (model.dat + scalers.dat + config.json)",
            data=zip_bytes,
            file_name="dwsim_ann_model.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # ── Full text report ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">📝 Full Text Report</div>', unsafe_allow_html=True)
        lines = [
            "=" * 60,
            "  DWSIM ANN Model Studio — Training Report",
            "=" * 60,
            f"  Dataset rows      : {total_n}",
            f"  Train / Val split : {train_n} / {val_n}",
            f"  Input features    : {', '.join(in_cols)}",
            f"  Output targets    : {', '.join(out_cols)}",
            f"  Hidden layers     : {hp['hidden']}",
            f"  Activation        : {hp['activation']}",
            f"  Learning rate     : {hp['lr']}",
            f"  Epochs            : {len(res['train_losses'])}",
            f"  Dropout           : {hp['dropout']}",
            f"  Batch norm        : {hp['use_bn']}",
            f"  Batch size        : {hp['batch_size']}",
            "",
            "-" * 60,
            "  VALIDATION METRICS",
            "-" * 60,
        ]
        for _, row in mdf_val.iterrows():
            lines.append(
                f"  {row['Output']:<25}  RMSE={row['RMSE']:.4f}  MAE={row['MAE']:.4f}  R²={row['R²']:.4f}  MAPE={row['MAPE (%)']:.2f}%"
            )
        lines += [
            "",
            f"  Overall avg R²   : {avg_r2:.4f}",
            f"  Overall avg RMSE : {avg_rmse:.4f}",
            f"  Final Train Loss : {train_final_loss:.6f}",
            f"  Final Val Loss   : {val_final_loss:.6f}",
            "=" * 60,
        ]
        report_text = "\n".join(lines)
        st.code(report_text, language="")
        st.download_button(
            "⬇️ Download Text Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="model_report.txt",
            mime="text/plain",
        )

# ── Tab 4: Predict ─────────────────────────────────────────────────────────────
with tab_predict:
    if not st.session_state.training_done or st.session_state.training_results is None:
        st.info("⚠️ Train the model first (go to **🚀 Train Model** tab).")
    else:
        res     = st.session_state.training_results
        in_cols = st.session_state.input_cols
        out_cols = st.session_state.output_cols

        st.markdown('<div class="section-header">🔮 Single-Point Prediction</div>', unsafe_allow_html=True)
        st.markdown("Enter input values to get instant model predictions:")

        inp_vals = {}
        n_col = min(len(in_cols), 3)
        cols = st.columns(n_col)
        for i, col in enumerate(in_cols):
            with cols[i % n_col]:
                col_data = df[col].dropna()
                default_val = float(col_data.mean()) if len(col_data) > 0 else 0.0
                inp_vals[col] = st.number_input(
                    col,
                    value=default_val,
                    format="%.4f",
                    key=f"pred_input_{col}",
                )

        if st.button("🔮 Predict", use_container_width=True):
            X_new = np.array([[inp_vals[c] for c in in_cols]], dtype=np.float32)
            Xn = res["scaler_x"].transform(X_new)
            Xt = torch.from_numpy(Xn)
            res["model"].eval()
            with torch.no_grad():
                yn = res["model"](Xt).numpy()
            y_pred_single = res["scaler_y"].inverse_transform(yn)[0]

            st.markdown('<div class="section-header">📤 Prediction Results</div>', unsafe_allow_html=True)
            pred_cols = st.columns(len(out_cols))
            for i, col in enumerate(out_cols):
                pred_cols[i].metric(col, f"{y_pred_single[i]:.5f}")

        st.markdown('<div class="section-header">📂 Batch Prediction from File</div>', unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload batch CSV/Excel (must have the same input columns)", type=["csv", "xlsx", "xls"], key="batch_upload")
        if batch_file:
            try:
                if batch_file.name.endswith((".xlsx", ".xls")):
                    batch_df = pd.read_excel(batch_file)
                else:
                    batch_df = pd.read_csv(batch_file)

                missing = [c for c in in_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"Missing columns in batch file: {missing}")
                else:
                    X_batch = batch_df[in_cols].values.astype(np.float32)
                    Xbn = res["scaler_x"].transform(X_batch)
                    Xbt = torch.from_numpy(Xbn)
                    res["model"].eval()
                    with torch.no_grad():
                        ybn = res["model"](Xbt).numpy()
                    y_batch_pred = res["scaler_y"].inverse_transform(ybn)
                    for i, col in enumerate(out_cols):
                        batch_df[f"pred_{col}"] = y_batch_pred[:, i]

                    st.success(f"✅ Predicted {len(batch_df)} rows.")
                    st.dataframe(batch_df.head(20), use_container_width=True)

                    csv_out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        data=csv_out,
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            except Exception as ex:
                st.error(f"Error processing batch file: {ex}")
