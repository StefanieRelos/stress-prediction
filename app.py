# ============================================================
#   Student Stress Level Prediction System — Flask App
#   Author: Stefanie S. Relos | BSIT-301
# ============================================================

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("stress_model.pkl")
stress_labels = ["High", "Low", "Medium"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age               = float(request.form["age"])
        gender            = request.form["gender"]
        sleep_hours       = float(request.form["sleep_hours"])
        screen_time       = float(request.form["screen_time"])
        study_hours       = float(request.form["study_hours"])
        physical_activity = request.form["physical_activity"]
        caffeine_intake   = float(request.form["caffeine_intake"])
        academic_pressure = request.form["academic_pressure"]

        gender_map   = {"Female": 0, "Male": 1}
        activity_map = {"High": 0, "Low": 1, "Medium": 2}
        pressure_map = {"High": 0, "Low": 1, "Medium": 2}

        input_data = pd.DataFrame([[
            age,
            gender_map.get(gender, 0),
            sleep_hours,
            screen_time,
            study_hours,
            activity_map.get(physical_activity, 1),
            caffeine_intake,
            pressure_map.get(academic_pressure, 0)
        ]], columns=[
            "age", "gender", "sleep_hours", "screen_time_hours",
            "study_hours", "physical_activity", "caffeine_intake",
            "academic_pressure"
        ])

        prediction_index = model.predict(input_data)[0]
        prediction_label = stress_labels[prediction_index]
        proba            = model.predict_proba(input_data)[0]
        confidence       = round(max(proba) * 100, 2)

        # --- Build dynamic reasons ---
        reasons = []
        if sleep_hours < 6:
            reasons.append(f"You only sleep <strong>{sleep_hours} hrs/day</strong> — below the recommended 7–9 hours. Sleep deprivation is a major trigger of mental and physical stress.")
        if screen_time > 6:
            reasons.append(f"Your screen time is <strong>{screen_time} hrs/day</strong> — too much screen exposure increases eye strain, disrupts melatonin, and worsens anxiety.")
        if study_hours > 8:
            reasons.append(f"You study <strong>{study_hours} hrs/day</strong> — over-studying without breaks leads to mental fatigue and burnout.")
        if caffeine_intake > 3:
            reasons.append(f"You consume <strong>{caffeine_intake} cups of caffeine/day</strong> — excessive caffeine disrupts sleep cycles and spikes cortisol (stress hormone).")
        if academic_pressure == "High":
            reasons.append("You reported <strong>High Academic Pressure</strong> — this is one of the strongest predictors of student stress.")
        if physical_activity == "Low":
            reasons.append("<strong>Low Physical Activity</strong> — exercise releases endorphins that naturally reduce stress. Without it, the body struggles to recover.")
        if not reasons:
            reasons.append("Your lifestyle data shows certain imbalances detected by the model across multiple factors combined.")

        form_data = {
            "age": age, "gender": gender,
            "sleep_hours": sleep_hours, "screen_time": screen_time,
            "study_hours": study_hours, "physical_activity": physical_activity,
            "caffeine_intake": caffeine_intake, "academic_pressure": academic_pressure
        }

        return render_template(
            "index.html",
            prediction=prediction_label,
            confidence=confidence,
            reasons=reasons,
            form_data=form_data,
            submitted=True
        )

    except Exception as e:
        return render_template("index.html", error=str(e), submitted=False)


if __name__ == "__main__":
    app.run(debug=True)