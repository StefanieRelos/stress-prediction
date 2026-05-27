# ============================================================
#   Student Stress Level Prediction System
#   With Authentication + CRUD (Predictions + Journal)
#   Author: Stefanie S. Relos | BSIT-301
# ============================================================

from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "stress_prediction_secret_2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///stress_app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please login to continue."

model = joblib.load("stress_model.pkl")
stress_labels = ["High", "Low", "Medium"]

# ============================================================
# MODELS
# ============================================================

class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.relationship("Prediction", backref="user", lazy=True, cascade="all, delete-orphan")
    journals      = db.relationship("Journal", backref="user", lazy=True, cascade="all, delete-orphan")

class Prediction(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    user_id           = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    age               = db.Column(db.Float)
    gender            = db.Column(db.String(10))
    sleep_hours       = db.Column(db.Float)
    screen_time       = db.Column(db.Float)
    study_hours       = db.Column(db.Float)
    physical_activity = db.Column(db.String(10))
    caffeine_intake   = db.Column(db.Float)
    academic_pressure = db.Column(db.String(10))
    stress_level      = db.Column(db.String(10))
    confidence        = db.Column(db.Float)
    created_at        = db.Column(db.DateTime, default=datetime.utcnow)

class Journal(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title      = db.Column(db.String(200), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    mood       = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================
# AUTH ROUTES
# ============================================================

@app.route("/")
def splash():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("splash.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not name or not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
        else:
            user = User(name=name, email=email, password_hash=generate_password_hash(password))
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash(f"Welcome, {user.name}! 🎉", "success")
            return redirect(url_for("dashboard"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("splash"))

# ============================================================
# PREDICTION ROUTES
# ============================================================

@app.route("/predict", methods=["GET"])
@login_required
def form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@login_required
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
            age, gender_map.get(gender, 0), sleep_hours, screen_time,
            study_hours, activity_map.get(physical_activity, 1),
            caffeine_intake, pressure_map.get(academic_pressure, 0)
        ]], columns=["age","gender","sleep_hours","screen_time_hours",
                     "study_hours","physical_activity","caffeine_intake","academic_pressure"])

        prediction_index = model.predict(input_data)[0]
        prediction_label = stress_labels[prediction_index]
        proba            = model.predict_proba(input_data)[0]
        confidence       = round(max(proba) * 100, 2)

        reasons = []
        if sleep_hours < 6:
            reasons.append(f"You only sleep <strong>{sleep_hours} hrs/day</strong> — below the recommended 7–9 hours.")
        if screen_time > 6:
            reasons.append(f"Your screen time is <strong>{screen_time} hrs/day</strong> — too much screen exposure worsens anxiety.")
        if study_hours > 8:
            reasons.append(f"You study <strong>{study_hours} hrs/day</strong> — over-studying leads to burnout.")
        if caffeine_intake > 3:
            reasons.append(f"You consume <strong>{caffeine_intake} cups of caffeine/day</strong> — disrupts sleep and spikes cortisol.")
        if academic_pressure == "High":
            reasons.append("You reported <strong>High Academic Pressure</strong> — strongest predictor of student stress.")
        if physical_activity == "Low":
            reasons.append("<strong>Low Physical Activity</strong> — exercise naturally reduces stress hormones.")
        if not reasons:
            reasons.append("Multiple lifestyle factors combined contributed to your stress level.")

        # Save to DB
        pred = Prediction(
            user_id=current_user.id, age=age, gender=gender,
            sleep_hours=sleep_hours, screen_time=screen_time,
            study_hours=study_hours, physical_activity=physical_activity,
            caffeine_intake=caffeine_intake, academic_pressure=academic_pressure,
            stress_level=prediction_label, confidence=confidence
        )
        db.session.add(pred)
        db.session.commit()

        session["prediction"]  = prediction_label
        session["confidence"]  = confidence
        session["reasons"]     = reasons
        session["pred_id"]     = pred.id
        session["form_data"]   = {
            "age": age, "gender": gender, "sleep_hours": sleep_hours,
            "screen_time": screen_time, "study_hours": study_hours,
            "physical_activity": physical_activity,
            "caffeine_intake": caffeine_intake, "academic_pressure": academic_pressure
        }
        return redirect(url_for("results"))

    except Exception as e:
        return render_template("index.html", error=str(e))

@app.route("/results")
@login_required
def results():
    prediction = session.get("prediction")
    if not prediction:
        return redirect(url_for("form"))
    return render_template("results.html",
        prediction=session.get("prediction"),
        confidence=session.get("confidence"),
        reasons=session.get("reasons"),
        form_data=session.get("form_data")
    )

# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.created_at.desc()).limit(5).all()
    journals    = Journal.query.filter_by(user_id=current_user.id)\
                    .order_by(Journal.created_at.desc()).limit(5).all()
    total_preds = Prediction.query.filter_by(user_id=current_user.id).count()
    return render_template("dashboard.html",
        predictions=predictions, journals=journals, total_preds=total_preds)

# ============================================================
# PREDICTION HISTORY CRUD
# ============================================================

@app.route("/history")
@login_required
def history():
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                    .order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)

@app.route("/history/delete/<int:pred_id>", methods=["POST"])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        flash("Unauthorized.", "error")
        return redirect(url_for("history"))
    db.session.delete(pred)
    db.session.commit()
    flash("Prediction deleted.", "success")
    return redirect(url_for("history"))

# ============================================================
# JOURNAL CRUD
# ============================================================

@app.route("/journal")
@login_required
def journal():
    journals = Journal.query.filter_by(user_id=current_user.id)\
                .order_by(Journal.created_at.desc()).all()
    return render_template("journal.html", journals=journals)

@app.route("/journal/add", methods=["GET", "POST"])
@login_required
def add_journal():
    if request.method == "POST":
        title   = request.form.get("title", "").strip()
        content = request.form.get("content", "").strip()
        mood    = request.form.get("mood", "Neutral")
        if not title or not content:
            flash("Title and content are required.", "error")
        else:
            j = Journal(user_id=current_user.id, title=title, content=content, mood=mood)
            db.session.add(j)
            db.session.commit()
            flash("Journal entry added! ✅", "success")
            return redirect(url_for("journal"))
    return render_template("journal_form.html", journal=None, action="Add")

@app.route("/journal/edit/<int:journal_id>", methods=["GET", "POST"])
@login_required
def edit_journal(journal_id):
    j = Journal.query.get_or_404(journal_id)
    if j.user_id != current_user.id:
        flash("Unauthorized.", "error")
        return redirect(url_for("journal"))
    if request.method == "POST":
        j.title      = request.form.get("title", "").strip()
        j.content    = request.form.get("content", "").strip()
        j.mood       = request.form.get("mood", "Neutral")
        j.updated_at = datetime.utcnow()
        if not j.title or not j.content:
            flash("Title and content are required.", "error")
        else:
            db.session.commit()
            flash("Journal updated! ✅", "success")
            return redirect(url_for("journal"))
    return render_template("journal_form.html", journal=j, action="Edit")

@app.route("/journal/delete/<int:journal_id>", methods=["POST"])
@login_required
def delete_journal(journal_id):
    j = Journal.query.get_or_404(journal_id)
    if j.user_id != current_user.id:
        flash("Unauthorized.", "error")
        return redirect(url_for("journal"))
    db.session.delete(j)
    db.session.commit()
    flash("Journal entry deleted.", "success")
    return redirect(url_for("journal"))

# ============================================================
# RUN
# ============================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)