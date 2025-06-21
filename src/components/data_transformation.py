import os
import sys
import numpy as np
import pandas as pd
from src.utils.logger import logger
from src.utils.exception import CustomException
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import joblib

@dataclass
class DataTransformationConfig:
    transformed_train_path: str = os.path.join("artifacts", "train_transformed.npy")
    transformed_test_path: str = os.path.join("artifacts", "test_transformed.npy")
    features_path: str = os.path.join("artifacts", "features.joblib")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        self.label_encoders = {}

    def preprocess(self, df: pd.DataFrame, fit: bool = True):
        try:
            df = df.copy()
            logger.info("🔧 Starting preprocessing...")

            # Drop ID column if exists
            df.drop(columns=['Loan_ID'], errors='ignore', inplace=True)

            # Fill missing values
            for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
                df[col].fillna(df[col].mode()[0], inplace=True)
            df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
            df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

            # Log transform
            df['LoanAmount_log'] = np.log1p(df['LoanAmount'])

            # Handle target if present
            if 'Loan_Status' in df.columns:
                df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

            # Label Encoding
            for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col])

            # Handle Dependents
            df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

            # Final features
            drop_cols = ['LoanAmount']
            if 'Loan_Status' in df.columns:
                drop_cols.append('Loan_Status')
            features = df.drop(columns=drop_cols).columns.tolist()

            logger.info(f"✅ Final features: {features}")
            return df[features].values, df['Loan_Status'].values if 'Loan_Status' in df.columns else None, features

        except Exception as e:
            logger.error("❌ Error during preprocessing")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train, y_train, features = self.preprocess(train_df, fit=True)
            X_test, y_test, _ = self.preprocess(test_df, fit=False)

            # Save transformed data
            np.save(self.config.transformed_train_path, np.c_[X_train, y_train])
            np.save(self.config.transformed_test_path, np.c_[X_test, y_test])

            # Save column names
            joblib.dump(features, self.config.features_path)

            logger.info("✅ Data transformation complete and saved.")
            return (
                self.config.transformed_train_path,
                self.config.transformed_test_path,
                self.config.features_path
            )

        except Exception as e:
            raise CustomException(e, sys)
