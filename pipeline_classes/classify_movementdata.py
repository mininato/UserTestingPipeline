import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from _config import config

# This class is used to classify the movement data using a pre-trained model
class ClassifyMovementData(BaseEstimator, TransformerMixin):
    def __init__(self, model_file = None):
        #self.model_path = model_path if model_path else config.get("model_path")
        self.model_file = model_file
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.model is None:
            if self.model_file is None:
                raise ValueError("Model file is not provided.")
            try:
                self.model = joblib.load(self.model_file)  # Load the model
            except Exception as e:
                raise ValueError(f"Failed to load the model file: {e}")

        # Assuming `X` is a DataFrame of pre-extracted features.
        predictions = self.model.predict(X)

        # Adding predictions to the DataFrame as the first column
        X.insert(0, 'predicted_emotion', predictions)

        print("Data classified successfully.")
        
        # Export the labeled DataFrame to CSV
        #window_length_str = str(config["window_length"])
        output_file = f"classified_movement_data.csv"
        X.to_csv(output_file, index=False)
        print(f"Classified movement data exported successfully to {output_file}.")

        return X 
