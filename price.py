import streamlit as st
import pandas as pd
import numpy as np
from models.price_optimizer import PriceOptimizer
from utils.data_processor import DataProcessor
from utils.visualization import Visualizer
import base64
import io
import zipfile

def load_css():
    with open('styles/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def create_download_zip():
    """Create a zip file containing project files"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add main.py
        with open("main.py", "r") as f:
            zf.writestr("main.py", f.read())

        # Add README.md
        with open("README.md", "r") as f:
            zf.writestr("README.md", f.read())

    return zip_buffer.getvalue()

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

    # Download section
    with st.sidebar:
        st.markdown("### Download Project Files")
        zip_data = create_download_zip()
        st.download_button(
            label="📥 Download Project Files",
            data=zip_data,
            file_name="price_optimization_project.zip",
            mime="application/zip",
            help="Download the project files including README.md and main.py"
        )

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
        st.info("Please upload your historical sales data to begin optimization")

if __name__ == "__main__":
    main()