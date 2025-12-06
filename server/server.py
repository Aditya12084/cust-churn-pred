# server/app.py
from flask import Flask, render_template, request, jsonify
import joblib
import json
from preprocess import transform_input

app = Flask(__name__,
            template_folder='../client',
            static_folder='../client')

# Loading model
with open('../model/best_rf_pipeline.pkl', 'rb') as f:
    model = joblib.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            raw_data = {
                'gender': request.form['query1'],
                'SeniorCitizen': int(request.form['query2']),
                'Partner': request.form['query3'],
                'Dependents': request.form['query4'],
                'tenure': float(request.form['query5']),
                'PhoneService': request.form['query6'],
                'MultipleLines': request.form['query7'],
                'InternetService': request.form['query8'],
                'OnlineSecurity': request.form['query9'],
                'OnlineBackup': request.form['query10'],
                'DeviceProtection': request.form['query11'],
                'TechSupport': request.form['query12'],
                'StreamingTV': request.form['query13'],
                'StreamingMovies': request.form['query14'],
                'Contract': request.form['query15'],
                'PaperlessBilling': request.form['query16'],
                'PaymentMethod': request.form['query17'],
                'MonthlyCharges': float(request.form['query18']),
                'TotalCharges': float(request.form['query19'] or 0)
            }

            X = transform_input(raw_data)

            churn_prob = model.predict_proba(X)[0][1]
            prob_percent = f"{churn_prob:.1%}"

            threshold = 0.6
            final_prediction = 1 if churn_prob >= threshold else 0

            if final_prediction == 1:
                risk_level = "HIGH RISK → WILL CHURN"
                color = "#ef4444"
                decision = "Customer will churn"
            else:
                risk_level = "LOW RISK → WILL STAY"
                color = "#10b981"
                decision = "Customer will stay"

            return render_template('index.html',
                                   probability=prob_percent,
                                   risk_level=risk_level,
                                   decision=decision,
                                   final_class=final_prediction,
                                   color=color)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)