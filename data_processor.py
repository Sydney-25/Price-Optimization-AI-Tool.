import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def load_data(self, file):
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def prepare_features(self, df):
        X = df.drop(columns=['Sales_Amount'])
        y = df['Sales_Amount']
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
