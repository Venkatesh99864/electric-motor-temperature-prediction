from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
transformer = joblib.load("transform.pkl")


# Home Page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction Page
@app.route("/predict")
def predict():
    return render_template("Manual_predict.html")


# Prediction
@app.route("/y_predict", methods=["POST"])
def y_predict():
    try:
        # Read form values
        values = [float(x) for x in request.form.values()]
        print("Input:", values)

        # Scale input
        transformed = transformer.transform([values])

        # Predict
        prediction = model.predict(transformed)[0]

        return render_template(
            "result.html",
            prediction=round(float(prediction), 2)
        )

    except Exception as e:
        print(e)
        return render_template(
            "result.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
