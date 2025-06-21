from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and column names
model = joblib.load("artifacts/model.joblib")
columns = joblib.load("artifacts/columns.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Convert categorical values to match training data
        gender = 1 if data['Gender'] == 'Male' else 0
        married = 1 if data['Married'] == 'Yes' else 0
        dependents = data['Dependents']
        if dependents == '3+':
            dependents = 3
        else:
            dependents = int(dependents)
        education = 1 if data['Education'] == 'Graduate' else 0
        self_employed = 1 if data['Self_Employed'] == 'Yes' else 0
        applicant_income = float(data['ApplicantIncome'])
        coapplicant_income = float(data['CoapplicantIncome'])
        loan_amount = float(data['LoanAmount'])
        loan_term = float(data['Loan_Amount_Term'])
        credit_history = int(data['Credit_History'])
        property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        property_area = property_area_map[data['Property_Area']]

        # Create input array
        loan_amount_log = np.log1p(loan_amount)

        final_input = np.array([[gender, married, dependents, education, self_employed,
                                 applicant_income, coapplicant_income, loan_term,
                                 credit_history, property_area, loan_amount_log]])

        prediction = model.predict(final_input)[0]
        prediction_text = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("result.html", prediction=prediction_text)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

