import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logger
from src.utils.exception import CustomException

class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "data.csv")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("📥 Starting data ingestion...")

        try:
            # Load raw dataset
            df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
            logger.info(f"✅ Raw data shape: {df.shape}")

            # Create artifacts folder if not exists
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info("📦 Raw data saved")

            # Split into train/test
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info("✅ Train/Test split completed")
            logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logger.error("❌ Error in data ingestion")
            raise CustomException(e, sys)
