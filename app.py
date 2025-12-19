from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# =========================
# LOAD MODEL & SCALER
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "diabetes_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# =========================
# ROUTE HOME
# =========================
@app.route("/")
def home():
    return render_template("home.html")

# =========================
# ROUTE DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# =========================
# ROUTE PREDICT
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodpressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"])
        ]

        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]

        if prediction == 1:
            result = "Berpotensi Diabetes"
        else:
            result = "Tidak Berpotensi Diabetes"

    return render_template("predict.html", result=result)

# =========================
# RUN APP (UNTUK LOCAL)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)