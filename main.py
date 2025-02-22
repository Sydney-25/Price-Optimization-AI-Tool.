import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Data Preparation (Embedded Data)

# Create a sample dataset (replace with your actual data)
data = {
    'Price': [134.49, 134.49, 134.49, 134.49, 134.49, 150, 150, 150, 150, 150, 160, 160, 160, 160, 160, 170, 170, 170, 170, 170],
    'Competitor_Price': [206.44, 158.01, 278.03, 222.66, 189.55, 220, 160, 280, 230, 190, 230, 170, 290, 240, 200, 240, 180, 300, 250, 210],
    'Item_Quantity': [435, 435, 435, 435, 435, 450, 450, 450, 450, 450, 460, 460, 460, 460, 460, 470, 470, 470, 470, 470],  # Example quantity
    'Sales_Amount_After_Discount': [11272.59, 11272.59, 11272.59, 11272.59, 11272.59, 12000, 12000, 12000, 12000, 12000, 12500, 12500, 12500, 12500, 12500, 13000, 13000, 13000, 13000, 13000] #Example [...]
}

df = pd.DataFrame(data)
# Feature Engineering (Example: Price Difference)
df['Price_Difference'] = df['Competitor_Price'] - df['Price']

# Define the target variable (e.g., Sales_Amount_After_Discount)
target = 'Sales_Amount_After_Discount'

# Define features for training
features = ['Price', 'Competitor_Price', 'Price_Difference', 'Item_Quantity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Step 2: Model Training (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Step 3: Optimization Function (Simplified)
def predict_sales(price, competitor_price, item_quantity, model):
    """Predict sales amount for a given price and competitor price."""
    price_difference = competitor_price - price
    input_data = pd.DataFrame([[price, competitor_price, price_difference, item_quantity]], columns=features)
    return model.predict(input_data)[0]


def optimize_price(competitor_price, item_quantity, model, price_range=(50, 300)):
    """Find the price that maximizes predicted sales within a given range."""
    best_price = None
    best_sales = -1  # Initialize with a very low value
    for price in np.arange(price_range[0], price_range[1], 1):  # Increment by 1
        sales = predict_sales(price, competitor_price, item_quantity, model)
        if sales > best_sales:
            best_sales = sales
            best_price = price
    return best_price, best_sales


# Step 4: Streamlit App

st.title("Optimal Price Recommendation")

# Input elements
st.header("Product Information")
price = st.number_input("Your Product's Price", value=100.0, min_value=0.0)
competitor_price = st.number_input("Competitor's Price", value=100.0, min_value=0.0)

default_item_quantity = int(df['Item_Quantity'].mean())
item_quantity = st.number_input("Item Quantity", value=default_item_quantity, min_value=1)  # Ensure at least 1 item

if st.button("Find Optimal Price"):
    # Find the optimal price
    optimal_price, predicted_sales = optimize_price(competitor_price, item_quantity, model)

    # Display the results
    if optimal_price is not None:
        st.success(f"Optimal Price: ${optimal_price:.2f}")
        st.info(f"Predicted Sales at Optimal Price: ${predicted_sales:.2f}")
    else:
        st.error("Could not find an optimal price.")


# Display Model Evaluation results
st.sidebar.header("Model Evaluation")
st.sidebar.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")  # Display RMSE on sidebar

# Explaining the model and the process
st.sidebar.header("Explanation")
st.sidebar.write("This app uses a Linear Regression model to predict sales based on your product's price, competitor's price, and item quantity. The model is trained on a small, embedded dataset. Provide a larger and more representative dataset for real-world scenarios.")
