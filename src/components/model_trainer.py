import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.utils.logger import logger
from src.utils.exception import CustomException
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.joblib")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            logger.info("🧠 Training model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            logger.info(f"✅ Accuracy: {acc:.4f}")
            logger.info("📄 Classification Report:\n" + classification_report(y_test, y_pred))

            return model, acc

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_path, test_path):
        try:
            logger.info("🚀 Loading transformed data...")
            train_arr = np.load(train_path)
            test_arr = np.load(test_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model, acc = self.train_and_evaluate(X_train, y_train, X_test, y_test)

            joblib.dump(model, self.config.model_path)
            logger.info(f"💾 Model saved to {self.config.model_path}")

            return self.config.model_path, acc

        except Exception as e:
            raise CustomException(e, sys)
