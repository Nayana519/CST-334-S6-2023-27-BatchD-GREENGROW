from flask import Flask, render_template, request, redirect
import joblib
import requests
import os
import numpy as np

# -------------------------------
# 1️⃣ INITIALIZE FLASK APP
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 2️⃣ LOAD MODEL AND ENCODERS SAFELY
# -------------------------------
model = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
le_crop = joblib.load("crop_encoder.pkl") if os.path.exists("crop_encoder.pkl") else None
le_fert = joblib.load("fertilizer_encoder.pkl") if os.path.exists("fertilizer_encoder.pkl") else None

# Replace with your real API key
OPENWEATHER_API_KEY = "YOUR_API_KEY"


# -------------------------------
# 3️⃣ WEATHER FUNCTION
# -------------------------------
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return temp, humidity, description
    except:
        # Default values if API fails
        return 25.0, 50.0, "Weather Unavailable"


# -------------------------------
# 4️⃣ ROUTES
# -------------------------------

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/login_submit", methods=["POST"])
def login_submit():
    return redirect("/")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Check if model exists
        if model is None:
            return "❌ model.pkl not found in project folder."

        # 🔹 Get Form Data
        N = float(request.form.get("Nitrogen", 0))
        P = float(request.form.get("Phosphorus", 0))
        K = float(request.form.get("Potassium", 0))
        pH = float(request.form.get("pH", 6.5))
        crop_name = request.form.get("Crop", "Wheat")
        city = request.form.get("City", "Mumbai")

        # 🔹 Get Weather Data
        temp, humidity, weather_desc = get_weather(city)

        # 🔹 Encode Crop (if encoder exists)
        if le_crop:
            try:
                crop_encoded = le_crop.transform([crop_name])[0]
            except:
                crop_encoded = 0
        else:
            crop_encoded = 0

        # 🔹 Default placeholder values
        moisture = 40.0
        soil_type = 0

        # 🔹 Feature Order:
        # [Temp, Humidity, Moisture, SoilType, CropType, Nitrogen, Potassium, Phosphorous]
        features = np.array([[temp, humidity, moisture, soil_type, crop_encoded, N, K, P]])

        # 🔹 Make Prediction
        prediction = model.predict(features)

        # 🔹 Decode Fertilizer
        if le_fert:
            fertilizer_name = le_fert.inverse_transform(prediction)[0]
        else:
            fertilizer_name = str(prediction[0])

        return render_template(
            "result.html",
            fertilizer=fertilizer_name,
            crop=crop_name,
            temp=temp,
            humidity=humidity,
            weather=weather_desc
        )

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"


# -------------------------------
# 5️⃣ RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
