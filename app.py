from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Load model
model = joblib.load("credit_model.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
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

# Run the app on a public IP/port required by Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
