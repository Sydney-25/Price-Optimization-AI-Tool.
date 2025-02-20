import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Or any other suitable regression model
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle  # To save and load the trained model
import os  # To check if model file exists


# --- Model Training and Saving (Executed only if model file doesn't exist) ---

MODEL_FILE = "price_optimization_model.pkl"

def train_model(df):
    """Trains a RandomForestRegressor model and saves it to a file."""

    try:  # Add this try block

        # Feature Engineering
        try:
            # Drop the column that causes the error
            df = df.drop('Fiscal_Week_ID', axis=1)
        except KeyError:
            st.error("Fiscal_Week_ID column not found in the DataFrame.")
            return None

        df['Store_ID'] = df['Store_ID'].astype('category').cat.codes
        df['Item_ID'] = df['Item_ID'].astype('category').cat.codes

        # Fill missing values (Important for robustness)
        df.fillna(df.mean(), inplace=True)

        # Define features and target
        features = ['Store_ID', 'Item_ID', 'Item_Quantity', 'Competition_Price']  # Removed Fiscal_Week_ID
        target = 'Sales_Amount'

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Root Mean Squared Error: {rmse}")
        st.session_state['model_rmse'] = rmse

        # Save the model
        with open(MODEL_FILE, "wb") as file:
            pickle.dump(model, file)

        print(f"Model saved to {MODEL_FILE}")  # Confirm save

        print("Model trained and saved successfully.")
        return model

    except Exception as e:  # Add this except block
        st.error(f"An error occurred during training: {e}")
        print(f"Detailed training error: {e}")  # Print to console for debugging
        return None



# --- Streamlit App ---

def load_data():
    """Loads data from uploaded CSV and preprocesses it."""
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Basic Validation - Add more robust validation for production.
            if df.empty:
                st.error("The CSV file is empty.")
                return None

            expected_columns = ['Store_ID', 'Item_ID', 'Price', 'Item_Quantity', 'Sales_Amount_No_Discount', 'Sales_Amount', 'Competition_Price'] #Removed Fiscal_Week_ID
            if not all(col in df.columns for col in expected_columns):
                st.error(f"The CSV file is missing required columns. Expected: {expected_columns}")
                return None

            # Ensure data types are appropriate (Very Important)
            try:
                df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
                df['Item_Quantity'] = pd.to_numeric(df['Item_Quantity'], errors='coerce')
                df['Sales_Amount_No_Discount'] = pd.to_numeric(df['Sales_Amount_No_Discount'], errors='coerce')
                df['Sales_Amount'] = pd.to_numeric(df['Sales_Amount'], errors='coerce')
                df['Competition_Price'] = pd.to_numeric(df['Competition_Price'], errors='coerce')
            except ValueError as e:
                st.error(f"Error converting columns to numeric: {e}")
                return None


            if 'Fiscal_Week_ID' in df.columns:  # Drop the column if it exists.
                df = df.drop('Fiscal_Week_ID', axis=1)
                st.warning("Fiscal_Week_ID column was dropped.")

            print("Data types after dropping Fiscal_Week_ID:")
            print(df.dtypes)  # Add this line


            return df

        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")
            return None
    else:
        st.warning("Please upload a CSV file.")
        return None


def load_model():
    """Loads the trained model from a file."""
    try:
        with open(MODEL_FILE, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None



def predict_sales(model, store_id, item_id, item_quantity, competition_price):  #Removed fiscal_week
    """Predicts sales amount given input parameters."""

    # Convert inputs to the correct format for the model
    input_data = pd.DataFrame({
        'Store_ID': [store_id],
        'Item_ID': [item_id],
        'Item_Quantity': [item_quantity],
        'Competition_Price': [competition_price]
    })


    # Convert categorical features to numeric using the same mapping as during training.  If you load a vocabulary file, use that here.
    input_data['Store_ID'] = input_data['Store_ID'].astype('category').cat.codes
    input_data['Item_ID'] = input_data['Item_ID'].astype('category').cat.codes


    prediction = model.predict(input_data)
    return prediction[0]


def optimize_price(model, store_id, item_id, item_quantity, competition_price, price_range, num_prices=10): #Removed fiscal_week
    """Finds the optimal price within a given range to maximize revenue."""
    prices = np.linspace(price_range[0], price_range[1], num_prices)
    revenues = []

    original_store_id = store_id
    original_item_id = item_id

    # Pre-encode
    store_id = pd.Series([store_id]).astype('category').cat.codes[0]
    item_id = pd.Series([item_id]).astype('category').cat.codes[0]


    for price in prices:

        sales = predict_sales(model, store_id, item_id, item_quantity, competition_price)  # Use encoded values #Removed fiscal_week
        revenue = price * sales
        revenues.append(revenue)

    optimal_price = prices[np.argmax(revenues)]
    max_revenue = np.max(revenues)

    # Decode (for display purposes only)
    store_id = original_store_id
    item_id = original_item_id

    return optimal_price, max_revenue



def main():
    st.title("Price Optimization App")

    if st.button("Clear Session State"):
        st.session_state.clear()
        st.rerun()  # Force a re-run of the app

    # Load data and model
    df = load_data()
    if df is None:
        st.stop()  # Stop execution if data loading failed

    # Check if the model file exists.  If not, train and save it.
    if not os.path.exists(MODEL_FILE):
        st.warning("Model file not found. Training the model...")
        model = train_model(df)
        if 'model_rmse' in st.session_state:  # Check if RMSE was calculated and stored.
            st.success(f"Model trained successfully.  RMSE: {st.session_state['model_rmse']:.2f}")
        else:
            st.warning("Model trained, but RMSE not available. Check training output.")


    model = load_model()
    if model is None:
        st.error("Failed to load the model.  Make sure the model file exists or that training completed successfully.")
        st.stop()



    # Input fields
    st.header("Enter Product Details")
    #fiscal_week = st.date_input("Fiscal Week", value=pd.to_datetime("2019-11-01"))  # Default value
    store_id = st.selectbox("Store ID", df['Store_ID'].unique())
    item_id = st.selectbox("Item ID", df['Item_ID'].unique())
    item_quantity = st.number_input("Item Quantity", min_value=1, value=100)
    competition_price = st.number_input("Competition Price", min_value=0.01, value=50.0)
    price_range = st.slider("Price Range", min_value=1.0, max_value=500.0, value=(50.0, 150.0)) # Provide a price range


    # Optimization button
    if st.button("Optimize Price"):
        with st.spinner("Optimizing price..."):
            try:
                optimal_price, max_revenue = optimize_price(model, store_id, item_id, item_quantity, competition_price, price_range)  # Added competition_price

                st.success(f"Optimal Price: ${optimal_price:.2f}")
                st.success(f"Predicted Maximum Revenue: ${max_revenue:.2f}")
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")



if __name__ == "__main__":
    main()
