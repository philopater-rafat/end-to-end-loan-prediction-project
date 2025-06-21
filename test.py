# test_predict.py

from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":
    input_data = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }

    pipeline = PredictPipeline()
    result = pipeline.predict(input_data)
    print(f"\n💡 Prediction Result: {result}")
