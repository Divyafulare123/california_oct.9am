from flask import Flask, request, render_template
import joblib

# Load model
obj = joblib.load("california.joblib")

model = obj["model"]
column = obj["columns"]

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        INPUT = []

        for i in column:
            value = request.form.get(i)

            if value is None or value.strip() == "":
                return render_template(
                    "index.html",
                    error=f"Please enter {i}"
                )

            INPUT.append(float(value))

        prediction = model.predict([INPUT])[0]

        return render_template(
            "index.html",
            prediction=f"{prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            error="Something went wrong. Please check your inputs."
        )


if __name__ == "__main__":
    app.run(debug=True)