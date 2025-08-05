from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")           # Trained model
transformer = joblib.load("transform.pkl") # Trained MinMaxScaler

# Home route
@app.route('/')
def index():
    return render_template('index.html')  # Landing/home page

# Prediction form route
@app.route('/predict')
def predict():
    return render_template('Manual_predict.html')  # Input form

# Prediction result route
@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        # Extract input values from form
        values = [float(x) for x in request.form.values()]
        print("📥 Input Values:", values)

        # Scale the values
        transformed = transformer.transform([values])
        print("🔧 Scaled Input:", transformed)

        # Predict
        prediction = model.predict(transformed)[0]
        print("📈 Prediction:", prediction)

        # Render result.html
        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        print("❌ Error:", e)
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
