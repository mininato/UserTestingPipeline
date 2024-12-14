import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from _config import config

# This class is used to classify the movement data using a pre-trained model
class ClassifyMovementData(BaseEstimator, TransformerMixin):
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else config.get("model_path")
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            self.model = joblib.load(self.model_path)  # Load the pre-trained model
            print(f"Model loaded from {self.model_path}")

        # Assuming `X` is a DataFrame of pre-extracted features.
        predictions = self.model.predict(X)

        # Adding predictions to the DataFrame as the first column
        X.insert(0, 'predicted_emotion', predictions)

        print("Data classified successfully.")
        
        # Export the labeled DataFrame to CSV
        window_length_str = str(config["window_length"])
        output_file = f"classified_movement_data_window_{window_length_str}.csv"
        X.to_csv(output_file, index=False)
        print(f"Classified movement data exported successfully to {output_file}.")

        return X 
