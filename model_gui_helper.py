"""
model_gui_helper.py

Standalone GUI tool to help build DWSIM external NN model packages.

Features:
- Let the user pick a CSV/Excel file.
- Show column names and let the user choose inputs and outputs.
- Optionally let the user tweak basic hyperparameters.
- Call the existing training/export code to generate:
    - model.dat
    - scalers.dat
    - config.json
    - A ZIP ready for DWSIM NN Model Wizard.

Run:
    python model_gui_helper.py
"""

import os
import sys
import json
import zipfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import threading

# Improve look-and-feel
try:
    style = ttk.Style()
    style.theme_use('clam')
except Exception:
    pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure we can import local modules (dwsim_like_ann, ModelExporter)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def select_file_dialog(title="Select file", filetypes=(("All files", "*.*"),)):
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return filepath


def save_folder_dialog(title="Select output folder"):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


class ColumnSelectorApp(tk.Tk):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.title("DWSIM NN Model Helper - Column Selector")
        self.geometry("800x500")

        self.df = df
        self.columns = list(df.columns)

        self.selected_inputs = []
        self.selected_outputs = []

        self._build_ui()

    def _build_ui(self):
        # Frames
        left_frame = tk.Frame(self)
        right_frame = tk.Frame(self)
        bottom_frame = tk.Frame(self)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Input columns list
        tk.Label(left_frame, text="Available Columns").pack()
        self.listbox_cols = tk.Listbox(left_frame, selectmode=tk.MULTIPLE)
        for col in self.columns:
            self.listbox_cols.insert(tk.END, col)
        self.listbox_cols.pack(fill=tk.BOTH, expand=True)

        # Input/Output selections
        tk.Label(right_frame, text="Inputs (X)").pack()
        self.listbox_inputs = tk.Listbox(right_frame)
        self.listbox_inputs.pack(fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="Outputs (Y)").pack()
        self.listbox_outputs = tk.Listbox(right_frame)
        self.listbox_outputs.pack(fill=tk.BOTH, expand=True)

        # Buttons in middle
        middle_frame = tk.Frame(self)
        middle_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        btn_add_input = ttk.Button(middle_frame, text=">> As Input", command=self.add_as_input)
        btn_add_output = ttk.Button(middle_frame, text=">> As Output", command=self.add_as_output)
        btn_clear = ttk.Button(middle_frame, text="Clear", command=self.clear_selection)

        btn_add_input.pack(pady=5)
        btn_add_output.pack(pady=5)
        btn_clear.pack(pady=5)

        # Bottom buttons
        btn_ok = ttk.Button(bottom_frame, text="Generate Model Package", command=self.on_ok)
        btn_cancel = ttk.Button(bottom_frame, text="Cancel", command=self.on_cancel)

        btn_ok.pack(side=tk.RIGHT, padx=5)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

    def add_as_input(self):
        selection = self.listbox_cols.curselection()
        for idx in selection:
            col = self.columns[idx]
            if col not in self.selected_inputs and col not in self.selected_outputs:
                self.selected_inputs.append(col)
                self.listbox_inputs.insert(tk.END, col)

    def add_as_output(self):
        selection = self.listbox_cols.curselection()
        for idx in selection:
            col = self.columns[idx]
            if col not in self.selected_outputs and col not in self.selected_inputs:
                self.selected_outputs.append(col)
                self.listbox_outputs.insert(tk.END, col)

    def clear_selection(self):
        self.selected_inputs.clear()
        self.selected_outputs.clear()
        self.listbox_inputs.delete(0, tk.END)
        self.listbox_outputs.delete(0, tk.END)

    def on_ok(self):
        if not self.selected_inputs or not self.selected_outputs:
            messagebox.showerror("Error", "Please select at least one input and one output.")
            return
        self.destroy()

    def on_cancel(self):
        self.selected_inputs = []
        self.selected_outputs = []
        self.destroy()

class SimpleMLP(nn.Module):
    """A small flexible MLP used by the GUI trainer."""
    def __init__(self, n_in, n_out, hidden_sizes, activation="tanh", dropout=0.0, use_bn=False):
        super().__init__()
        layers = []
        last = n_in
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
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
            else:
                layers.append(nn.Tanh())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def train_and_evaluate(df, input_cols, output_cols, hidden_layers, epochs, lr=1e-3, activation="tanh", dropout=0.0, use_bn=False, batch_size=32, callback=None, stop_event=None):
    # Extract numpy arrays
    X = df[input_cols].values.astype(np.float32)
    y = df[output_cols].values.astype(np.float32)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Scalers
    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_n = scaler_x.transform(X_train)
    X_val_n = scaler_x.transform(X_val)
    y_train_n = scaler_y.transform(y_train)
    y_val_n = scaler_y.transform(y_val)

    # Torch tensors
    X_train_t = torch.from_numpy(X_train_n)
    y_train_t = torch.from_numpy(y_train_n)
    X_val_t = torch.from_numpy(X_val_n)
    y_val_t = torch.from_numpy(y_val_n)

    model = SimpleMLP(
        n_in=X.shape[1],
        n_out=y.shape[1],
        hidden_sizes=hidden_layers,
        activation=activation,
        dropout=dropout,
        use_bn=use_bn,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        if stop_event is not None and stop_event.is_set():
            break

        model.train()
        opt.zero_grad()
        y_pred = model(X_train_t)
        loss = loss_fn(y_pred, y_train_t)
        loss.backward()
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            vloss = loss_fn(y_val_pred, y_val_t).item()

        train_losses.append(loss.item())
        val_losses.append(vloss)

        # denormalize predictions for callback (detach tensors first)
        y_train_pred_n = y_pred.detach().cpu().numpy()
        y_train_pred = scaler_y.inverse_transform(y_train_pred_n)
        y_val_pred_n = y_val_pred.detach().cpu().numpy()
        y_val_pred = scaler_y.inverse_transform(y_val_pred_n)

        # compute RMSE per output
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2, axis=0))

        if callback is not None:
            try:
                callback(epoch=epoch,
                         train_loss=float(loss.item()),
                         val_loss=float(vloss),
                         val_rmse=val_rmse,
                         y_train=y_train,
                         y_train_pred=y_train_pred,
                         y_val=y_val,
                         y_val_pred=y_val_pred,
                         history={"train_losses": train_losses, "val_losses": val_losses})
            except Exception:
                pass

    # Compute RMSE per output (on validation set, in original scale)
    model.eval()
    with torch.no_grad():
        y_val_pred_n = model(X_val_t).detach().cpu().numpy()
    y_val_pred = scaler_y.inverse_transform(y_val_pred_n)

    with torch.no_grad():
        y_train_pred_n = model(X_train_t).detach().cpu().numpy()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_n)

    rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2, axis=0))

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "rmse": rmse,
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
    }

    return model, scaler_x, scaler_y, metrics
def show_results_window(metrics, output_cols):
    # New Tk window
    win = tk.Toplevel()
    win.title("Training Results")

    # Text summary (RMSE)
    text = tk.Text(win, height=10)
    text.pack(fill=tk.X, padx=10, pady=5)

    text.insert(tk.END, "Validation RMSE per output:\n")
    for name, rm in zip(output_cols, metrics["rmse"]):
        text.insert(tk.END, f"  {name}: {rm:.4f}\n")

    text.insert(tk.END, "\n(Plot shows first output: actual vs predicted)\n")
    text.config(state=tk.DISABLED)

    # Matplotlib Figure
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    y_val = metrics["y_val"][:, 0]
    y_val_pred = metrics["y_val_pred"][:, 0]
    ax.plot(y_val, label="Actual", marker="o", linestyle="None")
    ax.plot(y_val_pred, label="Predicted", marker="x", linestyle="None")
    ax.set_xlabel("Sample")
    ax.set_ylabel(output_cols[0])
    ax.legend()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


class TrainingWindow(tk.Toplevel):
    """A window that shows live training progress (loss curves and preds)."""
    def __init__(self, parent, output_cols):
        super().__init__(parent)
        self.title("Live Training Monitor")
        self.geometry("900x600")
        self.output_cols = output_cols

        # controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(ctrl, text="Output to visualise:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=output_cols[0])
        self.output_menu = ttk.OptionMenu(ctrl, self.output_var, output_cols[0], *output_cols)
        self.output_menu.pack(side=tk.LEFT, padx=6)

        self.progress_label = ttk.Label(ctrl, text="Epoch: 0 | train_loss: - | val_loss: -")
        self.progress_label.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self._on_stop)
        self.btn_stop.pack(side=tk.RIGHT)

        # Figure
        self.fig, (self.ax_loss, self.ax_pred) = plt.subplots(2, 1, figsize=(7, 6))
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # data
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

        # queue for thread-safe updates
        self._queue = []
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()

    def _on_stop(self):
        self._stop_event.set()
        self.btn_stop.config(state=tk.DISABLED)

    def enqueue_update(self, **kwargs):
        # called from worker thread: only append to queue. DO NOT call tkinter methods here.
        with self._queue_lock:
            self._queue.append(kwargs)

    def _start_polling(self):
        # schedule periodic queue flush from main thread
        self._flush_queue()
        self.after(100, self._start_polling)

    def _flush_queue(self):
        with self._queue_lock:
            items = list(self._queue)
            self._queue.clear()
        for data in items:
            # handle errors queued by worker
            if isinstance(data, dict) and data.get('error'):
                messagebox.showerror('Training error', data.get('error'))
                continue
            self._apply_update(data)

    def _apply_update(self, data):
        epoch = data.get('epoch')
        train_loss = data.get('train_loss')
        val_loss = data.get('val_loss')
        y_train = data.get('y_train')
        y_train_pred = data.get('y_train_pred')
        y_val = data.get('y_val')
        y_val_pred = data.get('y_val_pred')

        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # update progress label
        self.progress_label.config(text=f"Epoch: {epoch} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        # loss plot
        self.ax_loss.clear()
        self.ax_loss.plot(self.epochs, self.train_losses, label='train')
        self.ax_loss.plot(self.epochs, self.val_losses, label='val')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('MSE Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        # predicted vs actual for selected output
        idx = self.output_cols.index(self.output_var.get())
        self.ax_pred.clear()
        # scatter val
        self.ax_pred.scatter(y_val[:, idx], y_val_pred[:, idx], label='val', alpha=0.7)
        # scatter train (smaller alpha)
        self.ax_pred.scatter(y_train[:, idx], y_train_pred[:, idx], label='train', alpha=0.4, marker='x')
        mn = min(y_val[:, idx].min(), y_val_pred[:, idx].min(), y_train[:, idx].min(), y_train_pred[:, idx].min())
        mx = max(y_val[:, idx].max(), y_val_pred[:, idx].max(), y_train[:, idx].max(), y_train_pred[:, idx].max())
        self.ax_pred.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        self.ax_pred.set_xlabel('Actual')
        self.ax_pred.set_ylabel(self.output_var.get())
        self.ax_pred.legend()

        self.canvas.draw()

    def start_training(self, df, input_cols, output_cols, hidden_layers, epochs, lr, activation, dropout, use_bn, out_dir, batch_size=32):
        # spawn worker thread
        def worker():
            try:
                model, scaler_x, scaler_y, metrics = train_and_evaluate(
                    df,
                    input_cols,
                    output_cols,
                    hidden_layers,
                    epochs,
                    lr=lr,
                    activation=activation,
                    dropout=dropout,
                    use_bn=use_bn,
                    batch_size=batch_size,
                    callback=self.enqueue_update,
                    stop_event=self._stop_event,
                )

                # final push of metrics if any
                final_epoch = len(metrics.get('train_losses', []))
                self.enqueue_update(epoch=final_epoch, train_loss=metrics['train_losses'][-1] if metrics['train_losses'] else 0.0,
                                    val_loss=metrics['val_losses'][-1] if metrics['val_losses'] else 0.0,
                                    y_train=metrics.get('y_train', np.zeros((1, len(output_cols)))),
                                    y_train_pred=metrics.get('y_train_pred', np.zeros((1, len(output_cols)))),
                                    y_val=metrics.get('y_val', np.zeros((1, len(output_cols)))),
                                    y_val_pred=metrics.get('y_val_pred', np.zeros((1, len(output_cols)))))

                # export model using ModelExporter if available
                try:
                    from ModelExporter import ModelExporter
                    exporter = ModelExporter(model, scaler_x, scaler_y)
                    exporter.export(str(out_dir))
                except Exception:
                    pass
            except Exception as e:
                # enqueue an error message so the main thread can show it
                with self._queue_lock:
                    self._queue.append({'error': str(e)})
                return

        # start polling loop in main thread to process queued updates
        self._start_polling()

        t = threading.Thread(target=worker, daemon=True)
        t.start()


class ParameterDialog(tk.Toplevel):
    """Dialog to collect training parameters with a nicer UI."""
    def __init__(self, parent, defaults=None):
        super().__init__(parent)
        self.title("Training Parameters")
        self.resizable(False, False)
        self.result = None
        if defaults is None:
            defaults = {}
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Learning rate:").grid(row=0, column=0, sticky='w')
        self.lr_var = tk.DoubleVar(value=defaults.get('lr', 1e-3))
        ttk.Entry(frm, textvariable=self.lr_var, width=20).grid(row=0, column=1, sticky='ew')

        ttk.Label(frm, text="Epochs:").grid(row=1, column=0, sticky='w')
        self.epochs_var = tk.IntVar(value=defaults.get('epochs', 150))
        ttk.Spinbox(frm, from_=1, to=10000, textvariable=self.epochs_var, width=18).grid(row=1, column=1, sticky='ew')

        ttk.Label(frm, text="Hidden layers (comma):").grid(row=2, column=0, sticky='w')
        self.hidden_var = tk.StringVar(value=defaults.get('hidden', '10'))
        ttk.Entry(frm, textvariable=self.hidden_var, width=20).grid(row=2, column=1, sticky='ew')

        ttk.Label(frm, text="Activation:").grid(row=3, column=0, sticky='w')
        self.act_var = tk.StringVar(value=defaults.get('activation', 'tanh'))
        acts = ['tanh', 'relu', 'leakyrelu', 'elu', 'selu', 'gelu', 'sigmoid']
        ttk.OptionMenu(frm, self.act_var, self.act_var.get(), *acts).grid(row=3, column=1, sticky='ew')

        ttk.Label(frm, text="Dropout:").grid(row=4, column=0, sticky='w')
        self.drop_var = tk.DoubleVar(value=defaults.get('dropout', 0.0))
        ttk.Entry(frm, textvariable=self.drop_var, width=20).grid(row=4, column=1, sticky='ew')

        self.bn_var = tk.BooleanVar(value=defaults.get('use_bn', False))
        ttk.Checkbutton(frm, text='Use BatchNorm', variable=self.bn_var).grid(row=5, column=0, columnspan=2, sticky='w')

        ttk.Label(frm, text="Batch size:").grid(row=6, column=0, sticky='w')
        self.bs_var = tk.IntVar(value=defaults.get('batch_size', 32))
        ttk.Spinbox(frm, from_=1, to=2048, textvariable=self.bs_var, width=18).grid(row=6, column=1, sticky='ew')

        btn_fr = ttk.Frame(frm)
        btn_fr.grid(row=7, column=0, columnspan=2, pady=8)
        ttk.Button(btn_fr, text='OK', command=self._on_ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_fr, text='Cancel', command=self._on_cancel).pack(side=tk.LEFT)

        self.protocol('WM_DELETE_WINDOW', self._on_cancel)

    def _on_ok(self):
        self.result = {
            'lr': float(self.lr_var.get()),
            'epochs': int(self.epochs_var.get()),
            'hidden': self.hidden_var.get(),
            'activation': self.act_var.get(),
            'dropout': float(self.drop_var.get()),
            'use_bn': bool(self.bn_var.get()),
            'batch_size': int(self.bs_var.get()),
        }
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

def generate_config_json(output_dir: Path, input_cols, output_cols, activation="tanh", hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [10]

    config = {
        "name": "Custom DWSIM ANN Model",
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

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def zip_model_folder(output_dir: Path, zip_name: str = None):
    if zip_name is None:
        zip_name = output_dir.name + ".zip"

    zip_path = output_dir.parent / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in ["config.json", "model.dat", "scalers.dat"]:
            fpath = output_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)

    return zip_path


def main():
    # 1) Ask for dataset file
    csv_path = select_file_dialog(
        title="Select training CSV/Excel file",
        filetypes=(
            ("Excel files", "*.xlsx;*.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ),
    )
    if not csv_path:
        return

    csv_path = Path(csv_path)

    # 2) Load with pandas
    try:
        if csv_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(csv_path)
        else:
            df = pd.read_csv(csv_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read file: {e}")
        return

    # 3) Ask user to choose columns (inputs/outputs)
    app = ColumnSelectorApp(df)
    app.mainloop()

    input_cols = app.selected_inputs
    output_cols = app.selected_outputs

    if not input_cols or not output_cols:
        # User cancelled or invalid
        return

    # 4) Ask for output directory
    out_dir = save_folder_dialog("Select output folder for model package")
    if not out_dir:
        return

    out_dir = Path(out_dir) / "exported_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Ask basic hyperparameters via a nicer dialog
    pd_defaults = {'lr': 1e-3, 'epochs': 150, 'hidden': '10', 'activation': 'tanh', 'dropout': 0.0, 'use_bn': False, 'batch_size': 32}
    param_dlg = ParameterDialog(None, defaults=pd_defaults)
    param_dlg.wait_window()
    if param_dlg.result is None:
        return

    epochs = param_dlg.result.get('epochs', 150)
    lr = param_dlg.result.get('lr', 1e-3)
    activation = param_dlg.result.get('activation', 'tanh')
    dropout = param_dlg.result.get('dropout', 0.0)
    use_bn = param_dlg.result.get('use_bn', False)
    batch_size = param_dlg.result.get('batch_size', 32)
    hidden_str = param_dlg.result.get('hidden', '10')
    try:
        hidden_layers = [int(x.strip()) for x in hidden_str.split(',') if x.strip()]
    except Exception:
        hidden_layers = [10]

    # 6) Train model directly from selected data
    try:
        from ModelExporter import ModelExporter
    except Exception as e:
        messagebox.showerror("Error", f"Failed to import ModelExporter: {e}")
        return

    try:
        messagebox.showinfo("Training", "Starting model training in a live monitor window...\nYou can stop training from the window.")

        # Launch live training window and run training in background
        root2 = tk.Tk()
        root2.withdraw()
        train_win = TrainingWindow(root2, output_cols)
        train_win.start_training(
            df=df,
            input_cols=input_cols,
            output_cols=output_cols,
            hidden_layers=hidden_layers,
            epochs=epochs,
            lr=lr,
            activation=activation,
            dropout=dropout,
            use_bn=use_bn,
            batch_size=batch_size,
            out_dir=out_dir,
        )
        # run Tk main loop so the live window is responsive
        root2.mainloop()

        # After the live window closes, assume training/export finished (or user stopped)
        # Generate config.json with selected column names
        generate_config_json(out_dir, input_cols, output_cols, activation=activation, hidden_layers=hidden_layers)

        # 7) Create ZIP for DWSIM
        zip_path = zip_model_folder(out_dir)

        messagebox.showinfo(
            "Success",
            f"Model trained and exported!\n\n"
            f"Folder: {out_dir}\n"
            f"ZIP for DWSIM: {zip_path}\n\n"
            f"You can now use this ZIP in the DWSIM NN Model Wizard.",
        )

    except Exception as e:
        messagebox.showerror("Error during training/export", f"An error occurred:\n{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
