import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualization import Visualizer
import base64
import io
import zipfile
from sklearn.linear_model import LinearRegression

class PriceOptimizer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def optimize_price(self, base_features, price_range):
        prices = np.linspace(price_range[0], price_range[1], 100)
        predicted_revenues = []

        for price in prices:
            features = base_features.copy()
            features[0] = price
            predicted_revenue = self.model.predict([features])[0]
            predicted_revenues.append(predicted_revenue)

        max_index = np.argmax(predicted_revenues)
        optimal_price = prices[max_index]
        max_revenue = predicted_revenues[max_index]

        return optimal_price, max_revenue, prices, predicted_revenues

def load_css():
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Price Optimization Tool",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    st.title("AI-Powered Price Optimization")
    st.markdown("Optimize your product pricing using machine learning")

    # Initialize components
    data_processor = DataProcessor()
    price_optimizer = PriceOptimizer()
    visualizer = Visualizer()

    # File upload
    uploaded_file = st.file_uploader("Upload your historical sales data (CSV)", type=['csv'])

    if uploaded_file:
        with st.spinner('Processing data...'):
            # Load and process data
            df = data_processor.load_data(uploaded_file)
            X, y = data_processor.prepare_features(df)
            X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

            # Train model
            price_optimizer.train(X_train, y_train)
            model_score = price_optimizer.evaluate(X_test, y_test)

            st.success(f"Model trained successfully! Accuracy: {model_score:.2f}")

        # Input section
        st.subheader("Price Optimization")
        col1, col2 = st.columns(2)

        with col1:
            current_price = st.number_input("Current Price", min_value=0.0, value=100.0)
            competition_price = st.number_input("Competition Price", min_value=0.0, value=120.0)

        with col2:
            min_price = st.number_input("Minimum Price", min_value=0.0, value=80.0)
            max_price = st.number_input("Maximum Price", min_value=0.0, value=150.0)

        if st.button("Optimize Price", key="optimize"):
            with st.spinner("Calculating optimal price..."):
                # Prepare features for optimization
                base_features = np.array([
                    current_price,
                    competition_price,
                    current_price/competition_price,
                    0,  # margin placeholder
                    0.8  # conversion rate placeholder
                ])

                # Get optimization results
                optimal_price, max_revenue, prices, revenues = price_optimizer.optimize_price(
                    base_features,
                    (min_price, max_price)
                )

                # Display results
                st.markdown(f"""
                <div class="card">
                    <h3>Optimization Results</h3>
                    <p class="success-text">Optimal Price: ${optimal_price:.2f}</p>
                    <p>Estimated Revenue: ${max_revenue:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                # Visualization
                price_plot = visualizer.create_price_optimization_plot(prices, revenues)
                st.plotly_chart(price_plot, use_container_width=True)

                # Competition analysis
                comp_analysis = visualizer.create_competition_analysis(
                    optimal_price,
                    df['Competition_Price']
                )
                st.plotly_chart(comp_analysis, use_container_width=True)

    else:
        # Create a dummy dataframe if no file is uploaded
        data = {'Price': [90, 100, 110, 120, 130],
                'Competition_Price': [110, 120, 130, 140, 150],
                'Sales': [50, 60, 70, 65, 55]}
        df = pd.DataFrame(data)

        # Process dummy data
        X, y = data_processor.prepare_features(df)
        X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

        # Train model
        price_optimizer.train(X_train, y_train)
        model_score = price_optimizer.evaluate(X_test, y_test)

        st.success(f"Model trained successfully with dummy data! Accuracy: {model_score:.2f}")

        # Input section
        st.subheader("Price Optimization")
        col1, col2 = st.columns(2)

        with col1:
            current_price = st.number_input("Current Price", min_value=0.0, value=100.0)
            competition_price = st.number_input("Competition Price", min_value=0.0, value=120.0)

        with col2:
            min_price = st.number_input("Minimum Price", min_value=0.0, value=80.0)
            max_price = st.number_input("Maximum Price", min_value=0.0, value=150.0)

        if st.button("Optimize Price", key="optimize"):
            with st.spinner("Calculating optimal price..."):
                # Prepare features for optimization
                base_features = np.array([
                    current_price,
                    competition_price,
                    current_price / competition_price,
                    0,  # margin placeholder
                    0.8  # conversion rate placeholder
                ])

                # Get optimization results
                optimal_price, max_revenue, prices, revenues = price_optimizer.optimize_price(
                    base_features,
                    (min_price, max_price)
                )

                # Display results
                st.markdown(f"""
                <div class="card">
                    <h3>Optimization Results</h3>
                    <p class="success-text">Optimal Price: ${optimal_price:.2f}</p>
                    <p>Estimated Revenue: ${max_revenue:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                # Visualization
                price_plot = visualizer.create_price_optimization_plot(prices, revenues)
                st.plotly_chart(price_plot, use_container_width=True)

                # Competition analysis
                comp_analysis = visualizer.create_competition_analysis(
                    optimal_price,
                    df['Competition_Price']
                )
                st.plotly_chart(comp_analysis, use_container_width=True)

if __name__ == "__main__":
    main()
