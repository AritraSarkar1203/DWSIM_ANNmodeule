# ⚗️ DWSIM ANN Model Studio

> **Train · Evaluate · Export · Deploy** — A fully interactive web application for building Neural Network models on chemical engineering datasets, with DWSIM-compatible binary export.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 What Is This?

This project provides **two ways** to train and export ANN models for DWSIM:

| Tool | Interface | Best For |
|------|-----------|----------|
| `streamlit_app.py` | 🌐 **Web App** (Streamlit) | Interactive use, sharing, live training charts, detailed report |
| `model_gui_helper.py` | 🖥️ Desktop GUI (Tkinter) | Local offline desktop use |
| `dwsim_like_ann.py` | ⌨️ Command Line | Scripted / automated training |

---

## 📁 Project Structure

```
ANN_Model/
│
├── streamlit_app.py          ← 🌟 NEW  Web App (main interface)
├── model_gui_helper.py       ← Desktop GUI (Tkinter)
├── dwsim_like_ann.py         ← Core training engine (CLI)
├── ModelExporter.py          ← Binary export for DWSIM C# wrapper
├── config_example.json       ← Sample configuration template
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## ✨ Streamlit Web App Features

| Feature | Description |
|---------|-------------|
| 📂 **Upload Dataset** | Drag & drop CSV or Excel files |
| 🔢 **Column Selector** | Pick input (X) and output (Y) columns interactively |
| ⚙️ **Hyperparameter Panel** | Learning rate, epochs, hidden layers, activation, dropout, batch norm |
| 📊 **Live Training** | Real-time progress bar + live loss curve during training |
| 📋 **Full Report** | RMSE, MAE, R², MAPE per output; parity plots; residual plots; error histogram |
| 🔍 **Feature Importance** | Permutation-based importance ranking |
| 🔮 **Predict** | Single-point prediction form + batch prediction from file upload |
| 📦 **Export** | Download DWSIM-compatible ZIP (`model.dat` + `scalers.dat` + `config.json`) |
| 📝 **Text Report** | Downloadable `.txt` summary of all metrics |

---

## 🚀 Quick Start — Run Locally

### Step 1 — Clone or download this repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME/ANN_Model
```

> If you downloaded as ZIP, extract it and open a terminal in the `ANN_Model` folder.

---

### Step 2 — Install Python dependencies

Make sure you have **Python 3.8 or higher** installed.

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit torch pandas numpy scikit-learn matplotlib openpyxl
```

> 💡 **Tip:** Use a virtual environment to avoid conflicts:
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # Mac / Linux
> pip install -r requirements.txt
> ```

---

### Step 3 — Launch the Web App

```bash
streamlit run streamlit_app.py
```

The app will open automatically at:

```
Local URL:   http://localhost:8501
Network URL: http://<your-ip>:8501
```

> Open your browser at **http://localhost:8501** if it doesn't open automatically.

---

### Step 4 — Use the App

1. **Upload** your CSV or Excel dataset from the **sidebar**
2. **Select** input columns (X) and output columns (Y)
3. **Adjust** hyperparameters (learning rate, epochs, architecture, etc.)
4. Go to **🚀 Train Model** tab → click **Start Training**
5. Watch the **live loss chart** update in real time
6. After training, go to **📋 Model Report** for the full analysis
7. **Download** the model ZIP for use in DWSIM

---

## 📦 requirements.txt

Create (or verify) a `requirements.txt` file in the `ANN_Model/` folder:

```
streamlit>=1.32.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
openpyxl>=3.1.0
```

---

###  Get your shareable link

Once deployed, Streamlit Cloud gives you a permanent public URL like:

```
https://dwsimannmodeule-vdjm4a8tk7sumwowjex5go.streamlit.app/
```

✅ **Share this link with anyone** — they can access the full app directly in their browser, no Python or installation required.

---

### 🔄 How to update the deployed app

Any time you push new code to GitHub, Streamlit Cloud **automatically redeploys**:

```bash
git add .
git commit -m "Improve model report charts"
git push
```

Streamlit Cloud picks up the change and re-deploys within ~1 minute.

---

## 🖥️ Desktop GUI (Alternative)

If you prefer a local desktop app instead of the web app, run:

```bash
python model_gui_helper.py
```

This opens a Tkinter GUI where you can:
- Pick a CSV/Excel dataset
- Choose input/output columns
- Set hyperparameters
- Watch live training charts
- Export model ZIP

---

## ⌨️ Command Line Training (Advanced)

For scripted / automated training:

```bash
python dwsim_like_ann.py \
    --data-csv your_data.xlsx \
    --hidden-sizes 64,32 \
    --activation selu \
    --epochs 300 \
    --lr 0.001 \
    --batch-size 64
```

Then export the trained model:

```bash
python ModelExporter.py
```

---

## 📊 Exported Model Format (DWSIM Compatible)

The downloaded ZIP contains three files that plug directly into DWSIM's Neural Network Unit Operation:

### `model.dat` — Binary weights & biases
```
[int32: layer count]
For each layer:
  [int32: output size]
  [int32: input size]
  [double[]: weights (row-major)]
  [double[]: biases]
```

### `scalers.dat` — Normalization parameters
```
[int32: input feature count]
[double[]: input mean]
[double[]: input scale]
[int32: output feature count]
[double[]: output mean]
[double[]: output scale]
```

### `config.json` — Model metadata
```json
{
  "name": "DWSIM ANN Model",
  "input_count": 4,
  "output_count": 2,
  "activation": "tanh",
  "model_file": "model.dat",
  "scaler_file": "scalers.dat",
  "hidden_layers": [64, 32],
  "input_names": ["T_in", "P_in", "F_in", "X_in"],
  "output_names": ["T_out", "P_out"]
}
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit: command not found` | Run `pip install streamlit` then retry |
| `ModuleNotFoundError: torch` | Run `pip install torch` |
| App opens but training crashes | Check that all required columns are selected in the sidebar |
| `ReduceLROnPlateau` error | Update PyTorch: `pip install --upgrade torch` |
| Streamlit Cloud deploy fails | Ensure `requirements.txt` exists in the repo and lists all packages |
| Blank page after deploy | Check the **Logs** in Streamlit Cloud dashboard for errors |

---

## 📚 Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [PyTorch](https://pytorch.org) | Neural network training |
| [scikit-learn](https://scikit-learn.org) | Preprocessing & metrics |
| [Matplotlib](https://matplotlib.org) | Charts & plots |
| [Pandas](https://pandas.pydata.org) | Data loading & manipulation |
| [NumPy](https://numpy.org) | Numerical computation |

---

## 📝 License

This project is open-source and integrates with [DWSIM](https://github.com/DanWBR/dwsim). Please follow DWSIM's licensing terms for any production DWSIM integration.

---

## 🌟 Contributing

Pull requests are welcome! If you find a bug or want a new feature:

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request on GitHub

---

<div align="center">

**Made with ❤️ for Chemical Engineers**


