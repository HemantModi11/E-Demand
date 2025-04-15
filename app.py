import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Delhi Peak Demand Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary: #3498db;
        --secondary: #2c3e50;
        --background: #f8f9fa;
        --card-bg: #0f0b50;
    }
    .main {
        background-color: var(--background);
    }
    .header {
        color: var(--secondary);
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #6c757d;
        font-size: 1.2em;
        margin-bottom: 1.5em;
    }
    .card {
        background: var(--card-bg);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5em;
        margin-bottom: 1.5em;
    }
    .feature-card {
        background: linear-gradient(135deg, #0f0b50 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1em;
        margin-bottom: 1em;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 10px;
        padding: 1.5em;
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .footer {
        color: #6c757d;
        font-size: 0.9em;
        text-align: center;
        margin-top: 2em;
        padding-top: 1em;
        border-top: 1px solid #e9ecef;
    }
    @media (max-width: 768px) {
        .header {
            font-size: 2em;
        }
        .card {
            padding: 1em;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = TemporalFusionTransformer.load_from_checkpoint("full_tft_model.pl", map_location=torch.device("cpu"))
        model.eval()
        return model

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None
model = load_model()

st.markdown('<div class="header">Delhi Peak Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict electricity peak demand using Temporal Fusion Transformer</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Interactive Forecast", "Next Week Forecast", "Data Insights"])

with tab1:
    st.markdown('<div class="card"><h3>Input Parameters</h3></div>', unsafe_allow_html=True)
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_date = st.date_input("Forecast Date", datetime.today() + timedelta(days=1))
            
            st.markdown('<div class="feature-card"><h4>Weather Conditions</h4></div>', unsafe_allow_html=True)
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
            humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 60.0, 1.0)
            precipitation = st.slider("Precipitation (mm)", 0.0, 100.0, 0.0, 0.1)
            
        with col2:
            st.markdown('<div class="feature-card"><h4>Environmental Factors</h4></div>', unsafe_allow_html=True)
            skin_temp = st.slider("Earth Skin Temperature (°C)", -20.0, 60.0, 25.0, 0.1)
            ev_count = st.number_input("Cumulative EVs in Delhi", min_value=0, value=50000, step=1000)
            
            last_peak = st.number_input("Last Observed Peak Demand (MW)", min_value=0.0, value=5000.0, step=100.0)
        
        submit_button = st.form_submit_button("Generate Forecast")

    if submit_button and model is not None:
        try:
            # Load full dataset (used for encoder window)
            df = pd.read_csv("final_data.csv", parse_dates=["Date"])
            df.columns = df.columns.str.strip().str.replace(" ", "_")
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["Day_of_week"] = df["Date"].dt.dayofweek
            df["group_id"] = "Delhi"
            df["time_idx"] = (df["Date"] - pd.Timestamp("2020-01-01")).dt.days

            # Selected forecast date
            selected_time_idx = (forecast_date - datetime(2020, 1, 1).date()).days

            # Hardcode encoder window to end on June 1, 2024 (or max date in CSV)
            last_encoder_date = df["Date"].max()  # Should be June 1, 2024
            encoder_time_idx = (last_encoder_date - pd.Timestamp("2020-01-01")).days
            encoder_df = df[df["time_idx"].between(encoder_time_idx - 55, encoder_time_idx)].copy()


            if encoder_df.shape[0] < 56:
                st.warning("Not enough historical data available for the selected forecast date.")
            else:
                # Create single forecast row
                forecast_row = {
                    "time_idx": selected_time_idx,
                    "group_id": "Delhi",
                    "Date": forecast_date,
                    "Month": forecast_date.month,
                    "Day": forecast_date.day,
                    "Day_of_week": forecast_date.weekday(),
                    "Temperature": temperature,
                    "Relative_Humidity": humidity,
                    "Precipitation": precipitation,
                    "Cumulative_EVs_Delhi": ev_count,
                    "Earth_Skin_Temperature": skin_temp,
                    "Peak_Demand": last_peak  # This will be overwritten by prediction
                }

                forecast_df = pd.DataFrame([forecast_row])

                # Merge encoder + decoder
                input_df = pd.concat([encoder_df, forecast_df], ignore_index=True)

                # Build dataset for prediction
                predict_dataset = TimeSeriesDataSet(
                    input_df,
                    time_idx="time_idx",
                    target="Peak_Demand",
                    group_ids=["group_id"],
                    min_encoder_length=56,
                    max_encoder_length=56,
                    min_prediction_length=1,
                    max_prediction_length=1,
                    time_varying_known_reals=["time_idx", "Month", "Day", "Day_of_week"],
                    time_varying_unknown_reals=[
                        "Temperature", "Relative_Humidity", "Precipitation",
                        "Cumulative_EVs_Delhi", "Earth_Skin_Temperature"
                    ],
                    target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    allow_missing_timesteps=True
                )

                predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)
                prediction = model.predict(predict_dataloader).cpu().numpy().flatten()[0]

                # Display result
                st.markdown('<div class="prediction-card"><h3>Forecast Results</h3></div>', unsafe_allow_html=True)
                st.metric("Predicted Peak Demand", f"{prediction:,.2f} MW")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

with tab2:
    st.markdown('<div class="card"><h3>Next Week Forecast</h3></div>', unsafe_allow_html=True)
    
    if st.button("Generate Next Week Forecast") and model is not None:
        try:
            # Load real training data
            df = pd.read_csv("final_data.csv", parse_dates=["Date"])
            df.columns = df.columns.str.strip().str.replace(" ", "_")

            # Add necessary features used in training
            df["Month"] = df["Date"].dt.month
            df["Day"] = df["Date"].dt.day
            df["Day_of_week"] = df["Date"].dt.dayofweek
            df["group_id"] = "Delhi"

            # Create time_idx
            df = df.sort_values("Date")
            df["time_idx"] = (df["Date"] - pd.Timestamp("2020-01-01")).dt.days

            max_encoder_length = 56
            max_prediction_length = 7

            max_time_idx = df["time_idx"].max()
            last_date = df["Date"].max()

            # Select encoder window (last 56 days)
            encoder_window = df[df["time_idx"] >= (max_time_idx - max_encoder_length + 1)].copy()

            # Create forecast horizon
            future_time_idxs = np.arange(max_time_idx + 1, max_time_idx + max_prediction_length + 1)
            forecast_rows = []
            for ti in future_time_idxs:
                new_date = last_date + pd.Timedelta(days=(ti - max_time_idx))
                forecast_rows.append({
                    "time_idx": ti,
                    "group_id": "Delhi",
                    "Date": new_date,
                    "Month": new_date.month,
                    "Day": new_date.day,
                    "Day_of_week": new_date.dayofweek,
                    "Temperature": encoder_window.iloc[-1]["Temperature"],
                    "Relative_Humidity": encoder_window.iloc[-1]["Relative_Humidity"],
                    "Precipitation": encoder_window.iloc[-1]["Precipitation"],
                    "Cumulative_EVs_Delhi": encoder_window.iloc[-1]["Cumulative_EVs_Delhi"],
                    "Earth_Skin_Temperature": encoder_window.iloc[-1]["Earth_Skin_Temperature"],
                    "Peak_Demand": encoder_window.iloc[-1]["Peak_Demand"]
                })

            forecast_df = pd.DataFrame(forecast_rows)

            # Merge encoder + forecast
            prediction_df = pd.concat([encoder_window, forecast_df], ignore_index=True)

            # Create dataset (same config as training)
            predict_dataset = TimeSeriesDataSet(
                prediction_df,
                time_idx="time_idx",
                target="Peak_Demand",
                group_ids=["group_id"],
                min_encoder_length=max_encoder_length,
                max_encoder_length=max_encoder_length,
                min_prediction_length=max_prediction_length,
                max_prediction_length=max_prediction_length,
                time_varying_known_reals=["time_idx", "Month", "Day", "Day_of_week"],
                time_varying_unknown_reals=[
                    "Temperature", "Relative_Humidity", "Precipitation",
                    "Cumulative_EVs_Delhi", "Earth_Skin_Temperature"
                ],
                target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )

            predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)
            predictions = model.predict(predict_dataloader).cpu().numpy().flatten()

            # Prepare results
            forecast_results = forecast_df[["Date", "time_idx"]].copy()
            forecast_results["Forecast_Peak_Demand (MW)"] = predictions

            # Display table
            st.dataframe(forecast_results[["Date", "Forecast_Peak_Demand (MW)"]].style.format({
                "Date": lambda x: x.strftime("%Y-%m-%d"),
                "Forecast_Peak_Demand (MW)": "{:,.2f}"
            }))

            # Plot
            fig = px.line(forecast_results, x="Date", y="Forecast_Peak_Demand (MW)",
                        title="Forecasted Peak Demand (June 2–9, 2024)",
                        template="plotly_white")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Forecasted Peak Demand (MW)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Forecast failed: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_time_series(df, cols, title="Time Series Plots"):
    fig = make_subplots(rows=len(cols)//2 + len(cols)%2, cols=2, subplot_titles=cols)
    row, col = 1, 1
    for y in cols:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[y], mode='lines', name=y),
                      row=row, col=col)
        col += 1
        if col > 2:
            row += 1
            col = 1
    fig.update_layout(height=300*row, title_text=title, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_interactive_distribution(df, cols):
    fig = make_subplots(rows=len(cols)//2 + len(cols)%2, cols=2, subplot_titles=cols)
    row, col = 1, 1
    for col_name in cols:
        fig.add_trace(go.Histogram(x=df[col_name], name=col_name, histnorm='probability density', opacity=0.75),
                      row=row, col=col)
        col += 1
        if col > 2:
            row += 1
            col = 1
    fig.update_layout(height=300*row, title_text="Distributions", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def plot_interactive_correlation(df):
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu', title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_boxplot(df, col='Peak_Demand'):
    fig = px.box(df, x='Month', y=col, title=f'Monthly Distribution of {col}')
    st.plotly_chart(fig, use_container_width=True)

def plot_lag_correlation(df, target_col, feature_col, max_lag=7):
    lags = range(1, max_lag+1)
    correlations = [df[target_col].corr(df[feature_col].shift(lag)) for lag in lags]
    fig = px.bar(x=list(lags), y=correlations, labels={'x': 'Lag', 'y': 'Correlation'},
                 title=f'Lag Correlation: {feature_col} vs {target_col}')
    st.plotly_chart(fig, use_container_width=True)

def plot_ev_vs_demand(df):
    fig = px.scatter(df, x='Cumulative_EVs_Delhi', y='Peak_Demand', color='Month',
                     title='EV Adoption vs Peak Demand')
    st.plotly_chart(fig, use_container_width=True)

def plot_scatter_matrix(df, cols):
    fig = px.scatter_matrix(df[cols], title="Scatter Matrix of Selected Features")
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def yearly_avg_plot(df, col='Peak_Demand'):
    yearly_avg = df.groupby('Year')[col].mean().reset_index()
    fig = px.bar(yearly_avg, x='Year', y=col, title=f'Yearly Average {col}')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="card"><h3>📊 Data Insights & Analytics</h3></div>', unsafe_allow_html=True)

    # Load and prepare data
    df = pd.read_csv("final_data.csv", parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    print(df.columns)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Time Series Plots
    st.subheader("📈 Time Series Trends")
    time_series_cols = ['Peak_Demand', 'Temperature', 'Cumulative_EVs_Delhi', 'Renewable_Energy_(%)']
    plot_interactive_time_series(df, time_series_cols)

    # Distributions
    st.subheader("📊 Distributions")
    distribution_cols = ['Peak_Demand', 'Temperature', 'Relative_Humidity', 'GDP_(Trillions_USD)']
    plot_interactive_distribution(df, distribution_cols)

    # Correlation
    st.subheader("🔗 Correlation Matrix")
    plot_interactive_correlation(df)

    # Monthly Boxplot
    st.subheader("📦 Monthly Variation")
    plot_monthly_boxplot(df)

    # EV vs Demand
    st.subheader("🔋 EV Adoption vs Peak Demand")
    plot_ev_vs_demand(df)

    # Scatter Matrix
    st.subheader("🧮 Scatter Matrix of Key Features")
    plot_scatter_matrix(df, ['Peak_Demand', 'Temperature', 'Cumulative_EVs_Delhi', 'Relative_Humidity'])

    # Yearly Average
    st.subheader("📆 Yearly Average Peak Demand")
    yearly_avg_plot(df)

    # Lag Correlation
    st.subheader("⏳ Lagged Correlation with Temperature")
    plot_lag_correlation(df, 'Peak_Demand', 'Temperature')

# Footer
st.markdown("""
<div class="footer">
    Delhi Power Demand Forecasting System | Hemant Modi, Sahil Makhija, Soumyadeep Roy Chowdhury, Dr. Yokesh Babu S<br>
    <small>Model Version: 1.0.0 | UI Version: 1.2.0</small>
</div>
""", unsafe_allow_html=True)