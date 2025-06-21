import os
import sys
import numpy as np
import pandas as pd
import joblib
from src.utils.exception import CustomException
from src.utils.logger import logger
from sklearn.preprocessing import LabelEncoder

class PredictPipeline:
    def __init__(self):
        try:
            self.model = joblib.load("artifacts/model.joblib")
            self.features = joblib.load("artifacts/features.joblib")
            logger.info("✅ Model and feature list loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def transform_input(self, data_dict):
        try:
            df = pd.DataFrame([data_dict])
            logger.info(f"🎯 Raw input: {df.to_dict(orient='records')}")

            # Handle missing optional keys
            for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
                if col in df:
                    df[col] = LabelEncoder().fit_transform(df[col])

            # Dependents
            df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

            # Log transform
            df['LoanAmount_log'] = np.log1p(df['LoanAmount'])

            # Drop raw LoanAmount
            df.drop(columns=['LoanAmount'], inplace=True)

            # Reorder columns
            df = df[self.features]

            logger.info("🔁 Transformed input ready for prediction.")
            return df.values

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: dict):
        try:
            data_arr = self.transform_input(input_data)
            prediction = self.model.predict(data_arr)[0]
            logger.info(f"✅ Prediction made: {prediction}")
            return "Approved ✅" if prediction == 1 else "Rejected ❌"
        except Exception as e:
            raise CustomException(e, sys)
