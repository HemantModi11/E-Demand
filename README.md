# ⚡ Delhi Peak Power Demand Forecasting

**Forecast the future of energy — today.**  
This project leverages advanced deep learning techniques to forecast Delhi's electricity peak demand, using rich weather, environmental, and economic indicators.

---

## 🔍 Project Overview

This platform predicts **Delhi’s daily peak power demand** using the Temporal Fusion Transformer (TFT), a powerful deep learning model built specifically for interpretable multi-horizon time series forecasting. The goal is to empower energy planners, analysts, and policymakers with accurate forecasts backed by both historical data and contextual insights.

---

## 🧠 Model Highlights

- **🌀 Temporal Fusion Transformer (TFT)**  
  Harnessing the power of attention mechanisms, recurrent layers, and embeddings to learn complex temporal patterns.

- **📊 Multi-variate Inputs**  
  Forecasts are based on historical trends and multiple influencing factors like:
  - Temperature
  - Humidity
  - Precipitation
  - Earth skin temperature
  - EV adoption rates
  - Macroeconomic indicators

- **📈 7-Day Ahead Forecasting**  
  The model predicts peak demand for the next 7 days using a 56-day encoder window of real historical data.

---

## 🌟 Key Features

### 🔮 Interactive Forecast
Users can input specific weather and environmental conditions for any future date within the forecast range (June 2–9, 2024) and get predicted peak demand instantly. This simulates “what-if” scenarios, such as extreme heatwaves or surges in EV usage.

### 📅 Automatic Weekly Forecast
Automatically generates the next 7 days of peak demand starting from the last available data (June 1, 2024), using the trained model and historical encoder window.

### 📊 Data Insights & Analytics
A dedicated dashboard provides rich visualizations of the dataset:
- Time series trends across key features
- Distributions and outlier views
- Correlation heatmaps
- Monthly and yearly demand patterns
- Scatter plots showing EV impact on demand
- Lag correlation between variables

These insights help users understand feature relationships and their effect on peak demand.

---

## ⚙️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for fast, interactive dashboards  
- **Backend Model**: [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) with Temporal Fusion Transformer  
- **Visualization**: [Plotly](https://plotly.com/) for interactive, publication-quality charts  
- **Data Analysis**: `pandas`, `numpy`, `statsmodels`  

---

## 📁 Data Summary

The forecasting model is trained on real daily data from **January 2020 to June 1, 2024**, including weather conditions, EV trends, and economic signals that influence electricity consumption.

---

## 🚀 Vision

As EV adoption, climate patterns, and urbanization evolve, the ability to **anticipate power demand accurately** becomes critical. This platform is a step toward making energy infrastructure smarter, more adaptive, and better informed by real-world data.
