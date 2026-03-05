
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# Minimal defaults so the module imports cleanly. Users should override
# these values when calling run_training or via CLI arguments.
DATA_CSV = "data.xlsx"
INPUT_COLS = []
OUTPUT_COLS = []
HIDDEN_SIZES = [64, 32]
ACTIVATION = "selu"
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 200

RELATIVE_MSE_TOLERANCE = 1e-3
ABSOLUTE_MSE_TOLERANCE = 1e-4

MODEL_SAVE = "dwsim_ann.pt"
SCALER_X_SAVE = "scaler_x.gz"
SCALER_Y_SAVE = "scaler_y.gz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model — Simple DWSIM-like MLP

class DWSIMLikeMLP(nn.Module):
    """Flexible MLP with optional BatchNorm and Dropout."""
    def __init__(self, n_in, n_out, hidden_sizes=[64, 32], activation="selu", dropout=0.0, use_bn=True):
        super().__init__()
        layers = []
        last = n_in

        for i, h in enumerate(hidden_sizes):
            lin = nn.Linear(last, h)
            layers.append(lin)

            if use_bn:
                layers.append(nn.BatchNorm1d(h))

            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leakyrelu":
                layers.append(nn.LeakyReLU(0.01))
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "selu":
                layers.append(nn.SELU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())

            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            last = h

        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)

        # weight init
        self._init_weights(activation)

    def _init_weights(self, activation):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "leakyrelu", "elu", "selu", "gelu"):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def run_training(
    data_csv: str = DATA_CSV,
    input_cols=None,
    output_cols=None,
    hidden_sizes=None,
    activation: str = ACTIVATION,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    train_fraction: float = 0.7,
    dropout: float = 0.1,
    use_bn: bool = True,
    patience: int = 25,
    rel_tol: float = RELATIVE_MSE_TOLERANCE,
    abs_tol: float = ABSOLUTE_MSE_TOLERANCE,
    model_save: str = MODEL_SAVE,
    scaler_x_save: str = SCALER_X_SAVE,
    scaler_y_save: str = SCALER_Y_SAVE,
):
    """Train the MLP and save model + scalers. Minimal validation of inputs.

    The function performs a train/validation split, normalization, training
    with early stopping and LR scheduling, and returns the trained model
    and scalers.
    """

    if not data_csv:
        raise ValueError("data_csv must be provided")
    if input_cols is None or output_cols is None or len(input_cols) == 0 or len(output_cols) == 0:
        raise ValueError("input_cols and output_cols must be provided as non-empty lists of column names")

    if hidden_sizes is None:
        hidden_sizes = HIDDEN_SIZES

    df = pd.read_excel(data_csv)
    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    # Split
    test_size = max(0.0, min(1.0, 1.0 - train_fraction))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Normalization
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_n = scaler_x.transform(X_train)
    X_val_n = scaler_x.transform(X_val)
    y_train_n = scaler_y.transform(y_train)
    y_val_n = scaler_y.transform(y_val)

    # Build model
    model = DWSIMLikeMLP(
        n_in=X.shape[1],
        n_out=y.shape[1],
        hidden_sizes=hidden_sizes,
        activation=activation,
        dropout=dropout,
        use_bn=use_bn,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_n), torch.from_numpy(y_train_n)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    print("Training ANN...")

    # training/validation history
    history = {"train_loss": [], "val_loss": [], "val_rmse": []}

    best_val_rmse = float('inf')
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0

        # Validation (compute denormalized RMSE)
        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val_n).to(DEVICE)
            val_pred_n = model(xv).cpu().numpy()

        # inverse scale
        val_pred = scaler_y.inverse_transform(val_pred_n)
        val_true = y_val
        val_rmse_per_output = np.sqrt(np.mean((val_pred - val_true) ** 2, axis=0))
        val_rmse_mean = float(np.mean(val_rmse_per_output))
        val_loss = float(np.mean((val_pred_n - y_val_n) ** 2))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse_mean)

        scheduler.step(val_loss)

        # early stopping / best model with tolerances
        improvement = best_val_rmse - val_rmse_mean
        rel_improvement = improvement / (best_val_rmse + 1e-12)

        is_better = False
        if best_val_rmse == float('inf'):
            is_better = True
        else:
            if abs_tol is not None and improvement > abs_tol:
                is_better = True
            if not is_better and rel_tol is not None and rel_improvement > rel_tol:
                is_better = True

        if is_better:
            best_val_rmse = val_rmse_mean
            best_state = model.state_dict()
            stale = 0
            torch.save(best_state, model_save)
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_rmse={val_rmse_mean:.6f}")

        if stale >= patience:
            print(f"Early stopping triggered (no improvement in {patience} epochs).")
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Denormalized Validation Metrics
    def predict(X_raw):
        Xn = scaler_x.transform(X_raw.astype(np.float32))
        Xn_t = torch.from_numpy(Xn).to(DEVICE)
        with torch.no_grad():
            yn = model(Xn_t).cpu().numpy()
        y = scaler_y.inverse_transform(yn)
        return y

    y_pred_val = predict(X_val)
    rmse = np.sqrt(np.mean((y_pred_val - y_val) ** 2, axis=0))
    print("\nValidation RMSE per output:", rmse)

    # Save Model & Scalers
    torch.save(model.state_dict(), model_save)
    joblib.dump(scaler_x, scaler_x_save)
    joblib.dump(scaler_y, scaler_y_save)
    print("\nSaved model and scalers.")

    return {
        'model': model,
        'scaler_x': scaler_x,
        'scaler_y': scaler_y,
        'history': history,
        'rmse': rmse,
    }


def parse_hidden_sizes(arg_hidden: str, n_hidden: int = None, first_neurons: int = None):
    if arg_hidden:
        try:
            return [int(x.strip()) for x in arg_hidden.split(',') if x.strip()]
        except Exception:
            raise ValueError('Invalid --hidden-sizes format. Use comma-separated integers, e.g. 160,80,40')
    if n_hidden and first_neurons:
        sizes = []
        val = int(first_neurons)
        for i in range(n_hidden):
            sizes.append(max(1, val))
            val = max(1, val // 2)
        return sizes
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DWSIM-like ANN with configurable hyperparameters')
    parser.add_argument('--data-csv', default=DATA_CSV)
    parser.add_argument('--hidden-sizes', type=str, help='Comma-separated hidden sizes, e.g. 160,80,40')
    parser.add_argument('--n-hidden', type=int, help='Number of hidden layers (used with --first-neurons)')
    parser.add_argument('--first-neurons', type=int, help='Neurons in first hidden layer when using --n-hidden')
    parser.add_argument('--activation', type=str, default=ACTIVATION)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no-bn', dest='use_bn', action='store_false', help='Disable batch normalization')
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--relative-mse-tol', type=float, default=RELATIVE_MSE_TOLERANCE)
    parser.add_argument('--absolute-mse-tol', type=float, default=ABSOLUTE_MSE_TOLERANCE)
    args = parser.parse_args()

    hs = parse_hidden_sizes(args.hidden_sizes, args.n_hidden, args.first_neurons)
    if hs is None:
        hs = HIDDEN_SIZES

    run_training(
        data_csv=args.data_csv,
        input_cols=INPUT_COLS,
        output_cols=OUTPUT_COLS,
        hidden_sizes=hs,
        activation=args.activation,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_fraction=args.train_fraction,
        dropout=args.dropout,
        use_bn=args.use_bn,
        patience=args.patience,
        rel_tol=args.relative_mse_tol,
        abs_tol=args.absolute_mse_tol,
        model_save=MODEL_SAVE,
        scaler_x_save=SCALER_X_SAVE,
        scaler_y_save=SCALER_Y_SAVE,
    )



