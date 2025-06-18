from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("credit_model.pkl")

# Define app
app = Flask(__name__)

# Define input endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    try:
        # Example: expecting exactly 20 features
        features = np.array([[
            data['Status'], data['Duration'], data['CreditHistory'], data['Purpose'],
            data['CreditAmount'], data['Savings'], data['EmploymentSince'],
            data['InstallmentRate'], data['PersonalStatusSex'], data['OtherDebtors'],
            data['ResidenceSince'], data['Property'], data['Age'],
            data['OtherInstallmentPlans'], data['Housing'], data['ExistingCredits'],
            data['Job'], data['NumLiablePeople'], data['Telephone'], data['ForeignWorker']
        ]])

        prediction = model.predict(features)[0]
        return jsonify({"risk": "high" if prediction == 0 else "low"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
