# prompt: /content/price_CSV.csv....generate a Streamlit Web application that meet the requirements of the Streamlit community cloud platform to train a ML model to determine the optimal price for a given product. The Streamlit Web application should have input elements for the price and competitor's price. The predictions should display the optimal price level of the given product that can maximize revenue while remaining competitive in the market.

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('/content/price_CSV.csv')

# Define features (X) and target (y)
X = df[['Competitor\'s_Price', 'Price']]  # Include both prices as features
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
st.write(f"Mean Squared Error: {mse}")

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

    # Determine optimal price (simple optimization based on predicted sales)
    # Optimize by slightly increasing the price or decrease if sales drop
    price_options = [price - 2, price -1, price, price + 1, price + 2]  # Adjust the price range
    best_price = price
    best_sales = predicted_sales

    for p in price_options:
        input_data = pd.DataFrame({'Competitor\'s_Price': [competitor_price], 'Price': [p]})
        predicted_sales = model.predict(input_data)[0]
        if predicted_sales > best_sales:
            best_price = p
            best_sales = predicted_sales
    
    st.write(f"Suggested Optimal Price to Maximize Revenue: ${best_price:.2f}")
