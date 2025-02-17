import numpy as np
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
        revenues = self.model.predict(prices.reshape(-1, 1) * base_features)
        max_revenue = np.max(revenues)
        optimal_price = prices[np.argmax(revenues)]
        return optimal_price, max_revenue, prices, revenues
