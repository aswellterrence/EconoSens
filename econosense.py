import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Mock Inflation Data (Replace with ZIMSTAT data if available)
inflation_data = {
    'Month': ['Jan-2023', 'Feb-2023', 'Mar-2023', 'Apr-2023', 'May-2023', 'Jun-2023',
              'Jul-2023', 'Aug-2023', 'Sep-2023', 'Oct-2023', 'Nov-2023', 'Dec-2023'],
    'Inflation_Rate': [25.6, 28.1, 30.5, 32.8, 35.0, 38.2, 40.5, 42.0, 43.8, 45.2, 47.5, 49.9]
}

# ARIMA Model for Inflation Forecasting
def predict_inflation():
    df = pd.DataFrame(inflation_data)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    model = ARIMA(df['Inflation_Rate'], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=3)  # Forecast next 3 months
    return forecast

# Monte Carlo Simulation for Cash Flow
def simulate_cash_flow(initial_cash, months, inflation):
    simulations = []
    for _ in range(1000):  # 1000 scenarios
        cash = initial_cash
        for _ in range(months):
            cash += np.random.normal(5000, 1000)  # Mock revenue
            cash -= np.random.normal(3000, 800)   # Mock costs
            cash *= (1 + inflation/100)  # Adjust for inflation
        simulations.append(cash)
    return simulations

# Streamlit App
st.title("📊 EconoSense MVP")
st.markdown("### Predictive Analytics for SMEs")

tab1, tab2 = st.tabs(["Inflation Forecast", "Cash Flow Simulator"])

with tab1:
    st.subheader("Zimbabwe Inflation Forecast")
    if st.button("Predict Next 3 Months"):
        forecast = predict_inflation()
        fig, ax = plt.subplots()
        ax.plot(forecast.index, forecast.values, marker='o', color='red')
        ax.set_xlabel("Month")
        ax.set_ylabel("Inflation Rate (%)")
        ax.set_title("Predicted Inflation (ARIMA Model)")
        st.pyplot(fig)
        st.write(f"**Next 3 Months Forecast:** {forecast.round(1).values}%")

with tab2:
    st.subheader("Cash Flow Simulation Under Inflation")
    initial_cash = st.number_input("Initial Cash (USD)", min_value=1000, value=10000)
    months = st.slider("Simulation Period (Months)", 1, 12, 6)
    inflation = st.slider("Expected Monthly Inflation (%)", 1.0, 10.0, 5.0)
    
    if st.button("Run Simulation"):
        simulations = simulate_cash_flow(initial_cash, months, inflation)
        fig, ax = plt.subplots()
        ax.hist(simulations, bins=30, color='green', alpha=0.7)
        ax.axvline(np.median(simulations), color='red', linestyle='dashed')
        ax.set_xlabel("Final Cash Position (USD)")
        ax.set_title("Monte Carlo Cash Flow Simulation")
        st.pyplot(fig)
        st.success(f"**Median Cash After {months} Months:** ${np.median(simulations):.2f}")

st.caption("Disclaimer: Mock data for demonstration. Real-world implementation requires SME financial data.")