# Cross-Market Prediction using Dynamic Neural Network

## 🚀 Project Overview

A sophisticated machine learning application for predicting cross-market volatility spillovers using Graph Neural Networks (GCN+GAT) and baseline MLP models.

## 📊 Live Application

**🌐 Web App:** https://iamrakii-cross-market-prediction.streamlit.app/

**Local Access:** `http://localhost:8501` (when running locally)

## 📁 Project Structure

```
├── project.py                 # Main data analysis & ML pipeline (709 lines)
├── streamlit_app.py          # Interactive web interface (412 lines)
├── run_app.py                # Application launcher
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── WEB_APP_GUIDE.md          # Web interface guide
├── GITHUB_PUSH_GUIDE.md      # GitHub deployment guide
├── templates/                # HTML templates
│   └── index.html
├── Data files (8 markets):
│   ├── ^GSPC_stock_data.csv  # S&P 500
│   ├── ^GDAXI_stock_data.csv # DAX
│   ├── ^FCHI_stock_data.csv  # CAC 40
│   ├── ^FTSE_stock_data.csv  # FTSE 100
│   ├── ^NSEI_stock_data.csv  # Nifty 50
│   ├── ^N225_stock_data.csv  # Nikkei 225
│   ├── ^KS11_stock_data.csv  # KOSPI
│   └── ^HSI_stock_data.csv   # Hang Seng
└── .venv/                    # Virtual environment

```

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.13 |
| **Data Processing** | Pandas, NumPy |
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Graph Neural Networks** | PyTorch Geometric (GCN, GAT) |
| **Time Series** | Statsmodels (VAR/FEVD) |
| **ML/Stats** | Scikit-learn, SciPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Data Source** | yfinance (Yahoo Finance) |
| **Network Analysis** | NetworkX |

## 📈 Key Features

### 1. **Data Analysis**
- Fetches 16 years of historical data (2007-2022) for 8 global stock indices
- Calculates realized volatility using squared returns
- Analyzes data quality and trading day statistics

### 2. **Spillover Analysis**
- Vector Autoregression (VAR) model with FEVD decomposition
- Diebold-Yilmaz spillover index computation
- Interactive heatmap visualization of market interconnections

### 3. **Graph Neural Network Model (GCN+GAT)**
- **Architecture:** GCNConv → GATConv → Output layers
- **Features:**
  - Hidden Dimension: 64
  - Attention Heads: 4
  - Layers: 3
  - Learning Rate: 0.001
  - Dropout: 0.3
- **Training:** Grid search with 71 epochs
- **Performance:** Best validation loss: 0.0421

### 4. **Baseline MLP Model**
- **Architecture:** 3 dense layers (128→64→32 neurons)
- **Training:** 50 epochs
- **Learning Rate:** 0.001
- **Dropout:** 0.5
- **Performance:** Best validation loss: 0.0451

### 5. **Volatility Forecasting**
- 22-day ahead predictions
- Market-specific forecasts
- Visual comparison with actual values

## 🌐 Web Application Pages

1. **Dashboard** - Overview of markets and data statistics
2. **Data Analysis** - Volatility spillover visualizations
3. **Spillover Analysis** - Market interconnection heatmaps
4. **Model Training** - GCN+GAT vs MLP comparison and training history
5. **Forecasting** - Volatility predictions with interactive controls
6. **About** - Project information and research background

## 🚀 Quick Start

### Local Setup
```powershell
# Navigate to project directory
cd "Cross-Market Prediction using Dynamic Neural Network"

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run web app
python -m streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud
```bash
# Ensure files are on GitHub
git push origin main

# Go to share.streamlit.io
# Select repository: iamrakii/cross-market-prediction
# Deploy streamlit_app.py
```

## 📦 Dependencies

All dependencies are in `requirements.txt`:
- numpy>=2.0.0
- pandas>=2.1.0
- torch>=2.1.0
- torch-geometric>=2.4.0
- streamlit>=1.28.0
- plotly>=5.17.0
- And 7 more...

Install with:
```bash
pip install -r requirements.txt
```

## 📊 Data Coverage

- **Time Period:** November 6, 2007 - June 3, 2022
- **Markets:** 8 global indices
- **Data Points:** ~3,800+ trading days per market
- **Data Quality:** >98%

## 🎯 Model Performance

| Model | Epochs | Best Loss | Training Time |
|-------|--------|-----------|---------------|
| GCN+GAT | 71 | 0.0421 | ~45 min |
| MLP | 50 | 0.0451 | ~12 min |

## 📝 Code Quality

- ✅ **Zero Errors:** All 23 type annotation issues resolved
- ✅ **Zero Warnings:** All unused imports removed
- ✅ **Syntax Valid:** Pylance verification passed
- ✅ **Compilation:** Python compilation test successful

## 📚 Documentation

- `README.md` - Project overview
- `WEB_APP_GUIDE.md` - Interactive interface guide
- `GITHUB_PUSH_GUIDE.md` - GitHub deployment instructions
- Inline code comments explaining key functions

## 🔗 GitHub Repository

**Repository:** https://github.com/iamrakii/cross-market-prediction

**Branch:** main

**Last Updated:** January 4, 2026

## 📧 Contact & Support

For questions or issues:
1. Check the documentation files
2. Review inline code comments
3. Visit GitHub issues section
4. Check Streamlit Cloud logs

## 📄 License

This project is open source and available for educational and research purposes.

---

**Project Status:** ✅ **COMPLETE & DEPLOYED**

- All code errors cleared
- Web app running on Streamlit Cloud
- GitHub repository up to date
- Ready for production use
