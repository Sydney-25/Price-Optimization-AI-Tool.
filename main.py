# prompt: Show the whole code script with brief explanations behind every step of the code execution.

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Load the dataset.  Error handling added for file not found.
try:
    df = pd.read_csv('price.csv') # Changed path to relative
except FileNotFoundError:
    st.error("Error: price.csv not found.  Make sure the file is in the same directory as the script, or upload it to /content.")
    st.stop() # Stop execution if file is not found

# Define features (X) and target (y)
X = df[['Competitor\'s_Price', 'Price']]
y = df['Sales_Amount_After_Discount']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Streamlit app
st.title("Price Optimization")
st.write("This app predicts the optimal product price to maximize revenue based on competitor pricing.")

# Input fields
price = st.number_input("Your Product Price:", min_value=0.0, value=50.0)
competitor_price = st.number_input("Competitor's Price:", min_value=0.0, value=60.0)

# Prediction
if st.button("Predict Optimal Price"):
    input_data = pd.DataFrame({'Competitor\'s_Price': [competitor_price], 'Price': [price]})
    predicted_sales = model.predict(input_data)[0]

    st.write(f"Predicted Sales Amount after Discount for this Price: ${predicted_sales:.2f}")

    # Determine optimal price
    price_options = [price - 2, price - 1, price, price + 1, price + 2]
    best_price = price
    best_sales = predicted_sales

    for p in price_options:
        input_data = pd.DataFrame({'Competitor\'s_Price': [competitor_price], 'Price': [p]})
        predicted_sales = model.predict(input_data)[0]
        if predicted_sales > best_sales:
            best_price = p
            best_sales = predicted_sales

    st.write(f"Suggested Optimal Price to Maximize Revenue: ${best_price:.2f}")
    st.write(f"Mean Squared Error: {mse}")
