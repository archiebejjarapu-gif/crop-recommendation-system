from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("crop_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values entered by the user
        values = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]

        # Prepare input data for the ML model
        columns = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        input_data = pd.DataFrame(
            [values],
            columns=columns
        )

        # Predict the suitable crop
        prediction = model.predict(input_data)[0]

        # Display the prediction on the webpage
        return render_template(
            "index.html",
            prediction=str(prediction)
        )

    except Exception as e:
        # Print the actual error in the terminal
        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            prediction=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)