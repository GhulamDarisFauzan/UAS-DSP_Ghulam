from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

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
        # Ambil data dari form (URUTAN HARUS SAMA DENGAN TRAINING)
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

        # Scaling data
        data_scaled = scaler.transform([data])

        # Prediksi
        prediction = model.predict(data_scaled)[0]

        # Hasil prediksi
        if prediction == 1:
            result = "Berpotensi Diabetes"
        else:
            result = "Tidak Berpotensi Diabetes"

    return render_template("predict.html", result=result)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
