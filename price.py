import streamlit as st
import pandas as pd
import numpy as np
from price_optimizer import PriceOptimizer
from data_processor import DataProcessor
from visualization import Visualizer

# Load CSS for custom styling
def load_css():
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cache the data loading function to optimize performance
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Cache the model training function
@st.cache_resource
def train_model(X_train, y_train):
    price_optimizer = PriceOptimizer()
    price_optimizer.train(X_train, y_train)
    return price_optimizer

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
    visualizer = Visualizer()

    # File upload
    uploaded_file = st.file_uploader("Upload your historical sales data (CSV)", type=['csv'])

    if uploaded_file:
        with st.spinner('Processing data...'):
            # Load and process data
            df = load_data(uploaded_file)
            if df is not None:
                # Ensure necessary columns are present
                required_columns = [
                    'Index', 'Fiscal_Week_ID', 'Store_ID', 'Item_ID', 
                    'Price', 'Item_Quantity', 'Sales_Amount_No_Discount', 
                    'Sales_Amount', 'Competition_Price'
                ]
                if not all(column in df.columns for column in required_columns):
                    st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
                    return

                X, y = data_processor.prepare_features(df)
                X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

                # Train model
                price_optimizer = train_model(X_train, y_train)
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
        st.info("Please upload your historical sales data to begin optimization")

if __name__ == "__main__":
    main()
