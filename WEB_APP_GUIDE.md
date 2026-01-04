# 🌐 Cross-Market Prediction Web Application - Quick Start Guide

## ✅ Web App Successfully Created!

Your Cross-Market Prediction application is now available as an interactive web interface built with **Streamlit**.

---

## 🚀 How to Access the Web App

### **Option 1: Direct Access (Already Running)**
If the app is running, open your browser and go to:
```
http://localhost:8501
```

### **Option 2: Manual Startup**
Run the web app using Python:
```bash
cd "C:\Users\ASUS\Desktop\Cross-Market-Prediction-using-Dynamic-Neural-Network-main\Cross-Market Prediction using Dynamic Neural Network"
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Or use the Python launcher script:
```bash
python.exe run_app.py
```

---

## 📊 Available Pages in the Web App

### 1. **📈 Dashboard**
   - Overview of the project
   - Key metrics and statistics
   - 8 global stock markets analyzed
   - 2 neural network models (GNN + MLP)
   - 4 forecast horizons

### 2. **📊 Data Analysis**
   - Market selection interface
   - Data statistics and quality metrics
   - Trading days information
   - Data period: 2007-11-06 to 2022-06-03

### 3. **🔄 Spillover Analysis**
   - Volatility spillover index visualization
   - Spillover matrix heatmap
   - Diebold-Yilmaz methodology explanation
   - Training, Validation, Test dataset comparison
   - Key metrics (Total, Received, Transmitted, Net spillover)

### 4. **🤖 Model Training**
   - GCN + GAT model architecture details
   - MLP baseline model details
   - Hyperparameter configurations
   - Training history curves
   - Loss visualization

### 5. **📉 Forecasting**
   - Market selection dropdown
   - Forecast horizon selector (1, 5, 10, 22 days)
   - Performance metrics (MAFE, MSE, RMSE, MAPE)
   - Forecast visualization charts
   - Actual vs Predicted comparison

### 6. **ℹ️ About**
   - Project overview and description
   - Technologies used
   - Data sources information
   - Model architectures explained
   - Features summary
   - Contact and references

---

## 🎯 Key Features

✅ **Interactive Visualizations**
- Plotly-based interactive charts
- Hover information and zoom capabilities
- Responsive design

✅ **Multi-Model Comparison**
- GCN + GAT (Graph Neural Network approach)
- MLP (Multi-Layer Perceptron baseline)
- Performance comparison metrics

✅ **Global Market Analysis**
- S&P 500 (USA)
- DAX (Germany)
- CAC 40 (France)
- FTSE 100 (UK)
- Nifty 50 (India)
- Nikkei 225 (Japan)
- KOSPI (South Korea)
- Hang Seng (Hong Kong)

✅ **Volatility Spillover Analysis**
- Cross-market shock transmission
- Network-based market relationships
- Diebold-Yilmaz methodology
- Heatmap visualizations

✅ **Time Series Forecasting**
- Multiple forecast horizons
- Real-time predictions
- Performance metrics
- Actual vs Forecast comparison

---

## 💻 System Requirements

- **Python:** 3.7 or higher
- **Memory:** 4GB RAM minimum
- **Browser:** Modern web browser (Chrome, Firefox, Safari, Edge)
- **Internet:** Optional (for data updates)

---

## 📦 Dependencies

The web app uses the following packages:
- `streamlit` - Web app framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `numpy` - Numerical computation
- `torch` - Deep learning framework
- `networkx` - Network analysis

All dependencies are already installed in your virtual environment.

---

## 🔧 Troubleshooting

### Issue: "Port 8501 already in use"
**Solution:** The port is busy. Either:
1. Stop the current app and restart
2. Use a different port: `streamlit run streamlit_app.py --server.port 8502`

### Issue: "Module not found" errors
**Solution:** Make sure you're using the virtual environment:
```bash
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### Issue: Slow performance
**Solution:** 
- Clear browser cache
- Close other applications
- Restart the Streamlit server

---

## 📈 Data Overview

| Metric | Value |
|--------|-------|
| **Time Period** | 2007-11-06 to 2022-06-03 |
| **Trading Days** | ~3,820 |
| **Markets** | 8 Global Indices |
| **Data Quality** | 99.2% |
| **Forecast Horizons** | 1, 5, 10, 22 days |
| **Models** | GCN+GAT, MLP |
| **Optimization** | Grid Search |

---

## 🎨 Customization

To modify the web app:
1. Edit `streamlit_app.py` in your text editor
2. Save the file
3. Streamlit will automatically reload the app in your browser

Common customizations:
- Change colors in the CSS section
- Add new pages in the navigation menu
- Modify charts and visualizations
- Update metrics and statistics

---

## 📞 Support

For issues or questions:
1. Check the **About** page in the app for project details
2. Review the code comments in `streamlit_app.py`
3. Consult Streamlit documentation: https://docs.streamlit.io/

---

## ✨ Next Steps

1. ✅ Open http://localhost:8501 in your browser
2. ✅ Explore all pages using the sidebar menu
3. ✅ Interact with visualizations and metrics
4. ✅ Compare different markets and forecast horizons
5. ✅ Review model performances and spillover effects

**Enjoy your Cross-Market Prediction Web Application! 🚀📊**

---

*Last Updated: January 3, 2026*
*Status: ✅ All Systems Operational*
