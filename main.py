import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

class DataProcessor:
    def load_data(self, file):
        """
        Load data from a CSV file.
        """
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def prepare_features(self, df: pd.DataFrame):
        """
        Clean the data and prepare features for model training.
        """
        # Clean the data by converting non-numeric values to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()  # Drop rows with NaN values

        # Check if the dataset is empty after cleaning
        if df.empty:
            st.error("The dataset is empty after cleaning. Please check your data.")
            return None, None

        X = df.drop(columns=['Sales_Amount'])
        y = df['Sales_Amount']
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        """
        # Ensure the dataset is not empty before splitting
        if X is None or y is None:
            return None, None, None, None

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

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
            if df is not None:
                X, y = data_processor.prepare_features(df)

                if X is not None and y is not None:
                    X_train, X_test, y_train, y_test = data_processor.split_data(X, y)

                    if X_train is not None and y_train is not None:
                        # Train model
                        price_optimizer.train(X_train, y_train)
                        model_score = price_optimizer.evaluate(X_test, y_test)

                        st.success(f"Model trained successfully! Accuracy: {model_score:.2f}")
                    else:
                        st.error("Failed to split the data. Please check your dataset.")
                else:
                    st.error("Failed to prepare features. Please check your dataset.")
