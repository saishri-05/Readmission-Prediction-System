import sqlite3
import json
import os
import sqlite3

import joblib
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for flash messages


# Function to check user credentials in the database
def check_user(email, password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user[5], password):  # user[4] is the hashed password in the database
        return user
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = check_user(email, password)
        
        if user:
            session['user_id'] = user[0]  # Store user ID in session
            session['name'] = user[1]  # Store user name in session
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/home')
def home():
    if 'user_id' in session:
        user_name = session.get('name', 'User')  # Get the user name from session
        return render_template('home.html', name=user_name)
    else:
        return redirect(url_for('login'))


def insert_user(name, age, email, mobile_number, password):
    # Use pbkdf2:sha256 for hashing the password
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Users (name, age, email, mobile_number, password) VALUES (?, ?, ?, ?, ?)",
                   (name, age, email, mobile_number, hashed_password))
    conn.commit()
    conn.close()



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        mobile_number = request.form['mobile_number']
        password = request.form['password']

        # Insert user into the database
        insert_user(name, age, email, mobile_number, password)
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')





@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')





conn = sqlite3.connect('database.db')
cursor = conn.cursor()




# ---------------- Load Models ---------------- #
STRESS_MODEL_PATH = "stress_model.pkl"
STRESS_LE_PATH = "label_encoder.pkl"
DISEASE_MODEL_PATH = "disease_model.pkl"
MLB_PATH = "mlb.pkl"

LOS_MODEL_PATH = "length_of_stay_model.pkl"
READM_MODEL_PATH = "readmission_30d_model.pkl"

xgb_stress = joblib.load(STRESS_MODEL_PATH)
stress_le = joblib.load(STRESS_LE_PATH)
disease_model = joblib.load(DISEASE_MODEL_PATH)
mlb = joblib.load(MLB_PATH)

los_model = joblib.load(LOS_MODEL_PATH) if os.path.exists(LOS_MODEL_PATH) else None
readm_model = joblib.load(READM_MODEL_PATH) if os.path.exists(READM_MODEL_PATH) else None

STRESS_FEATURES = ["steps_per_day", "calorie_intake", "smoking", "alcohol", "sleep_pattern"]

DISEASE_FEATURES = [
    "age","gender","rbc_count","wbc_count","systole","diastole","heart_rate","allergies",
    "blood_sugar","bmi","cholesterol","glucose_mean","hemoglobin_mean","length_of_stay",
    "steps_per_day","calorie_intake","smoking","alcohol","sleep_pattern","num_claims"
]

CLAIM_FEATURES = ["service_type","diagnosis_code","procedure_code","cost_of_service"]

# ---------------- DB ---------------- #
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS CheckupData (
        checkup_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        age INTEGER, gender INTEGER, rbc_count REAL, wbc_count REAL,
        systole INTEGER, diastole INTEGER, heart_rate INTEGER, allergies INTEGER,
        blood_sugar REAL, bmi REAL, cholesterol REAL,
        glucose_mean REAL, hemoglobin_mean REAL,
        steps_per_day INTEGER, calorie_intake INTEGER, smoking INTEGER, alcohol INTEGER,
        sleep_pattern INTEGER, num_claims INTEGER,
        service_type TEXT, diagnosis_code TEXT, procedure_code TEXT, cost_of_service REAL,
        procedure_codes TEXT,
        FOREIGN KEY(user_id) REFERENCES Users(user_id)
    );""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ResultData (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        checkup_id INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        stress_level TEXT,
        diseases TEXT,
        risk_score REAL,
        predicted_los REAL,
        readmission_prob REAL,
        readmission_label TEXT,
        FOREIGN KEY(user_id) REFERENCES Users(user_id),
        FOREIGN KEY(checkup_id) REFERENCES CheckupData(checkup_id)
    );""")
    conn.commit()
    conn.close()

init_db()

# ---------------- Prediction Helpers ---------------- #
def predict_stress_level(steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern):
    X = pd.DataFrame([{
        "steps_per_day": steps_per_day,
        "calorie_intake": calorie_intake,
        "smoking": smoking,
        "alcohol": alcohol,
        "sleep_pattern": sleep_pattern,
    }], columns=STRESS_FEATURES)
    enc_pred = xgb_stress.predict(X)[0]
    label = stress_le.inverse_transform([enc_pred])[0]
    return label, int(enc_pred)

def predict_diseases(patient_row_dict):
    X = pd.DataFrame([{k: patient_row_dict.get(k, None) for k in DISEASE_FEATURES}], columns=DISEASE_FEATURES)
    yhat = disease_model.predict(X)
    labels = mlb.inverse_transform(yhat)
    return list(labels[0]) if labels else []

def calculate_risk_category(stress_level, disease_count, length_of_stay, readmission_prob):
    stress_factor = stress_level / 2
    disease_factor = min(disease_count / 5, 1)
    los_factor = min(length_of_stay / 30, 1)
    readmission_factor = readmission_prob
    risk = (0.25 * stress_factor + 0.25 * disease_factor + 0.25 * los_factor + 0.25 * readmission_factor)
    risk_score = round(risk * 10, 2)
    if risk_score <= 4:
        category = "Low Risk"
    elif risk_score <= 7:
        category = "Medium Risk"
    else:
        category = "High Risk"
    return risk_score, category

def predict_claim_outcomes(claim_dict):
    if not (los_model and readm_model):
        return None, None, None
    Xc = pd.DataFrame([{
        "service_type": claim_dict.get("service_type"),
        "diagnosis_code": claim_dict.get("diagnosis_code"),
        "procedure_code": claim_dict.get("procedure_code"),
        "cost_of_service": float(claim_dict.get("cost_of_service", 0.0) or 0.0),
    }], columns=CLAIM_FEATURES)
    los = float(los_model.predict(Xc)[0])
    proba = readm_model.predict_proba(Xc)[0][1]
    label = "Yes" if proba >= 0.5 else "No"
    return round(los, 2), float(proba), label

# ---------------- Routes ---------------- #
@app.route("/checkup")
def checkup():
    if "user_id" not in session:
        session["user_id"] = 1
        session["name"] = "User"
    return render_template("checkup.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_id = session.get("user_id")
    user_name = session.get("name", "User")

    f = request.form.get
    age, gender = int(f("age")), int(f("gender"))
    rbc_count, wbc_count = float(f("rbc_count")), float(f("wbc_count"))
    systole, diastole = int(f("systole")), int(f("diastole"))
    heart_rate, allergies = int(f("heart_rate")), int(f("allergies"))
    blood_sugar, bmi, cholesterol = float(f("blood_sugar")), float(f("bmi")), float(f("cholesterol"))

    # Handle optional fields that might not be present in all forms
    glucose_mean = float(f("glucose_mean") or blood_sugar)  # Use blood_sugar as fallback
    hemoglobin_mean = float(f("hemoglobin_mean") or 14.0)   # Use default hemoglobin value

    steps_per_day, calorie_intake = int(f("steps_per_day")), int(f("calorie_intake"))
    smoking, alcohol, sleep_pattern = int(f("smoking")), int(f("alcohol")), int(f("sleep_pattern"))
    num_claims = int(f("num_claims") or 0)
    service_type, diagnosis_code, procedure_code = f("service_type") or "checkup", f("diagnosis_code") or "Z00", f("procedure_code") or "P01"
    cost_of_service = float(f("cost_of_service") or 150.0)
    procedure_codes = f("procedure_codes") or procedure_code

    # Stress
    stress_label, stress_enc = predict_stress_level(steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern)

    # Patient data
    patient_row = {
        "age": age, "gender": gender, "rbc_count": rbc_count, "wbc_count": wbc_count,
        "systole": systole, "diastole": diastole, "heart_rate": heart_rate, "allergies": allergies,
        "blood_sugar": blood_sugar, "bmi": bmi, "cholesterol": cholesterol,
        "glucose_mean": glucose_mean, "hemoglobin_mean": hemoglobin_mean,
        "length_of_stay": 0,
        "steps_per_day": steps_per_day, "calorie_intake": calorie_intake,
        "smoking": smoking, "alcohol": alcohol, "sleep_pattern": sleep_pattern,
        "num_claims": num_claims, "procedure_code": procedure_code, "procedure_codes": procedure_codes
    }

    predicted_los, readmission_prob, readmission_label = predict_claim_outcomes({
        "service_type": service_type,
        "diagnosis_code": diagnosis_code,
        "procedure_code": procedure_code,
        "cost_of_service": cost_of_service
    })
    if predicted_los is not None:
        patient_row["length_of_stay"] = predicted_los

    diseases = predict_diseases(patient_row)
    risk_score, risk_category = calculate_risk_category(stress_enc, len(diseases), patient_row["length_of_stay"], readmission_prob or 0)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO CheckupData
        (user_id, age, gender, rbc_count, wbc_count, systole, diastole, heart_rate, allergies,
         blood_sugar, bmi, cholesterol, glucose_mean, hemoglobin_mean,
         steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern,
         num_claims, service_type, diagnosis_code, procedure_code, cost_of_service, procedure_codes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
    """, (user_id, age, gender, rbc_count, wbc_count, systole, diastole, heart_rate, allergies,
          blood_sugar, bmi, cholesterol, glucose_mean, hemoglobin_mean,
          steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern,
          num_claims, service_type, diagnosis_code, procedure_code, cost_of_service, procedure_codes))
    checkup_id = cur.lastrowid

    diseases_str = ", ".join(diseases) if diseases else "Healthy"
    cur.execute("""
        INSERT INTO ResultData
        (user_id, report_id, created_at, stress_level, diseases, risk_score, predicted_los, readmission_prob, readmission_label)
        VALUES (?, ?, datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?)
    """, (user_id, checkup_id, stress_label, diseases_str, risk_score, predicted_los, readmission_prob, readmission_label))
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        name=user_name,
        stress_level=stress_label,
        diseases=diseases,
        risk_score=risk_score,
        risk_category=risk_category,
        predicted_los=predicted_los,
        readmission_prob=readmission_prob,
        readmission_label=readmission_label
    )

@app.route('/my_reports')
def my_reports():
    user_id = session.get('user_id')

    if not user_id:
        flash("Please log in to view your reports.")
        return redirect(url_for('login'))

    # Connect to the database and retrieve the user's checkup data with analysis results
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT c.created_at, c.checkup_id, c.age, c.gender, c.bmi,
               r.stress_level, r.diseases, r.risk_score, r.predicted_los,
               r.readmission_prob, r.readmission_label
        FROM CheckupData c
        JOIN ResultData r ON r.report_id = c.checkup_id
        WHERE c.user_id = ?
        ORDER BY c.created_at DESC
    ''', (user_id,))
    checkup_data = cursor.fetchall()
    conn.close()

    return render_template('my_reports.html', checkup_data=checkup_data)



@app.route('/previous_results')
def previous_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session.get('user_id')

    # Connect to the database and retrieve the user's result data with date
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT created_at, report_id, stress_level, diseases, risk_score
        FROM ResultData WHERE user_id = ?
        ORDER BY created_at DESC
    ''', (user_id,))
    results = cursor.fetchall()
    conn.close()

    return render_template('previous_results.html', results=results)



@app.route('/my_profile', methods=['GET', 'POST'])
def my_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        email = request.form.get('email')
        mobile_number = request.form.get('mobile_number')

        # Validate required fields
        if not name or not age or not email:
            flash('Please fill in all required fields.', 'error')
        else:
            try:
                # Update user's profile in the database
                cursor.execute('''
                    UPDATE Users
                    SET name = ?, age = ?, gender = ?, email = ?, mobile_number = ?
                    WHERE id = ?
                ''', (name, age, gender, email, mobile_number, user_id))
                conn.commit()

                # Update session name if changed
                session['name'] = name

                flash('Profile updated successfully!', 'success')
            except Exception as e:
                flash('Error updating profile. Please try again.', 'error')

    # Retrieve the current user's information
    cursor.execute('SELECT name, age, gender, email, mobile_number FROM Users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()

    # Get user's health statistics
    health_stats = get_user_health_stats(user_id, cursor)

    # Get user's activity summary
    activity_summary = get_user_activity_summary(user_id, cursor)

    conn.close()

    return render_template('my_profile.html',
                         user_data=user_data,
                         health_stats=health_stats,
                         activity_summary=activity_summary)

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session.get('user_id')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # Validate inputs
        if not current_password or not new_password or not confirm_password:
            flash('Please fill in all fields.', 'error')
            return render_template('change_password.html')

        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('change_password.html')

        if len(new_password) < 6:
            flash('New password must be at least 6 characters long.', 'error')
            return render_template('change_password.html')

        # Verify current password
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT password FROM Users WHERE id = ?', (user_id,))
        user_data = cursor.fetchone()

        if not user_data or not check_password_hash(user_data[0], current_password):
            flash('Current password is incorrect.', 'error')
            conn.close()
            return render_template('change_password.html')

        # Update password
        try:
            hashed_new_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            cursor.execute('UPDATE Users SET password = ? WHERE id = ?', (hashed_new_password, user_id))
            conn.commit()
            conn.close()

            flash('Password changed successfully!', 'success')
            return redirect(url_for('my_profile'))
        except:
            flash('Error changing password. Please try again.', 'error')
            conn.close()
            return render_template('change_password.html')

    return render_template('change_password.html')




@app.route('/diet_plan', methods=['GET', 'POST'])
def diet_plan():
    user_name = session.get('name', 'User')
    user_id = session.get('user_id')

    if request.method == 'POST':
        # Get form data
        age = int(request.form.get('age'))
        height = float(request.form.get('height'))
        weight = float(request.form.get('weight'))
        food_type = request.form.get('food_type')
        diseases = request.form.getlist('diseases')
        activity_level = request.form.get('activity_level', 'moderate')

        # Calculate BMI
        bmi = weight / ((height/100) ** 2)

        # Get user's latest health data if available
        health_data = get_user_health_data(user_id) if user_id else {}

        # Generate personalized diet plan
        diet_plan_data = generate_diet_plan(age, height, weight, bmi, food_type, diseases, activity_level, health_data)

        return render_template('diet_plan_result.html',
                             name=user_name,
                             diet_plan=diet_plan_data,
                             bmi=round(bmi, 1),
                             age=age,
                             weight=weight,
                             height=height)

    return render_template('diet_plan.html', name=user_name)


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        message = request.form.get('message')

        # Validate required fields
        if not name or not email or not mobile or not message:
            flash('Please fill in all required fields.', 'error')
            return render_template('feedback.html')

        # Store feedback in database
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()

            # Create feedback table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    mobile TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    status TEXT DEFAULT 'pending'
                )
            ''')

            # Insert feedback
            user_id = session.get('user_id')
            cursor.execute('''
                INSERT INTO Feedback (user_id, name, email, mobile, message)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, name, email, mobile, message))

            conn.commit()
            conn.close()

            flash('Thank you for your feedback! We will get back to you soon.', 'success')
            return redirect(url_for('thank_you'))

        except Exception as e:
            print(e," is the error")
            flash('An error occurred while submitting your feedback. Please try again.', 'error')
            return render_template('feedback.html')

    return render_template('feedback.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/view_feedback')
def view_feedback():
    """Admin route to view all feedback submissions"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, email, mobile, message, created_at, status
            FROM Feedback
            ORDER BY created_at DESC
        ''')
        feedback_list = cursor.fetchall()
        conn.close()

        return render_template('view_feedback.html', feedback_list=feedback_list)
    except:
        flash('Error loading feedback data.', 'error')
        return redirect(url_for('home'))


@app.route('/preventive_measures', methods=['GET', 'POST'])
def preventive_measures():
    if request.method == 'POST':
        # Get the form data
        risk_level = int(request.form.get('risk_level'))
        diseases = request.form.getlist('diseases')

        # Generate preventive measures based on risk level and diseases
        preventive_measures = generate_preventive_measures(risk_level, diseases)

        return render_template('preventive_measures_result.html', preventive_measures=preventive_measures)

    return render_template('preventive_measures.html')

def generate_preventive_measures(risk_level, diseases):
    measures = []

    # General recommendations based on risk level
    if risk_level >= 4:
        measures.append("Follow a regular exercise routine suitable to your health.")
        measures.append("Consult with a healthcare provider regularly.")
    elif risk_level == 3:
        measures.append("Engage in moderate physical activity.")
        measures.append("Maintain a balanced diet with fruits and vegetables.")
    elif risk_level <= 2:
        measures.append("Ensure a healthy lifestyle with balanced nutrition and regular activity.")

    # Disease-specific preventive measures
    if 'Diabetes' in diseases:
        measures.append("Monitor your blood glucose levels.")
        measures.append("Incorporate fiber-rich foods into your meals.")

    if 'Heart Disease' in diseases:
        measures.append("Avoid saturated fats and cholesterol.")
        measures.append("Limit alcohol consumption.")

    if 'Hypertension' in diseases:
        measures.append("Reduce sodium intake in your diet.")
        measures.append("Increase potassium intake through fruits and vegetables.")

    if 'Liver Disease' in diseases:
        measures.append("Avoid alcohol and substances harmful to the liver.")
        measures.append("Consult your doctor before taking any new medications.")

    if 'Asthma' in diseases:
        measures.append("Avoid allergens and pollutants.")
        measures.append("Maintain a healthy weight.")

    if 'Obesity' in diseases:
        measures.append("Follow a calorie-controlled diet.")
        measures.append("Incorporate regular physical activity.")

    if 'Anemia' in diseases:
        measures.append("Incorporate iron-rich foods like lean meats and spinach.")
        measures.append("Consume vitamin C-rich foods to enhance iron absorption.")

    # Additional combined disease management recommendations
    if 'Diabetes' in diseases and 'Heart Disease' in diseases:
        measures.append("Follow a heart-healthy, low-sugar diet.")

    if 'Hypertension' in diseases and 'Heart Disease' in diseases:
        measures.append("Follow a DASH diet and reduce stress.")

    if 'Diabetes' in diseases and 'Obesity' in diseases:
        measures.append("Focus on a weight management plan with low-impact exercises.")

    return measures

def get_user_health_data(user_id):
    """Get user's latest health data from the database"""
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT age, gender, bmi, blood_sugar, cholesterol, systole, diastole,
                   heart_rate, steps_per_day, calorie_intake, smoking, alcohol
            FROM CheckupData
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                'age': result[0],
                'gender': result[1],
                'bmi': result[2],
                'blood_sugar': result[3],
                'cholesterol': result[4],
                'systole': result[5],
                'diastole': result[6],
                'heart_rate': result[7],
                'steps_per_day': result[8],
                'calorie_intake': result[9],
                'smoking': result[10],
                'alcohol': result[11]
            }
    except:
        pass
    return {}

def get_user_health_stats(user_id, cursor):
    """Get user's health statistics and trends"""
    try:
        # Get latest health data
        cursor.execute('''
            SELECT bmi, blood_sugar, cholesterol, systole, diastole, heart_rate
            FROM CheckupData
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))
        latest_health = cursor.fetchone()

        # Get total checkups count
        cursor.execute('SELECT COUNT(*) FROM CheckupData WHERE user_id = ?', (user_id,))
        total_checkups = cursor.fetchone()[0]

        # Get latest risk assessment
        cursor.execute('''
            SELECT stress_level, risk_score, diseases
            FROM ResultData r
            JOIN CheckupData c ON r.report_id = c.checkup_id
            WHERE c.user_id = ?
            ORDER BY r.created_at DESC
            LIMIT 1
        ''', (user_id,))
        latest_risk = cursor.fetchone()

        return {
            'latest_health': latest_health,
            'total_checkups': total_checkups,
            'latest_risk': latest_risk
        }
    except:
        return {
            'latest_health': None,
            'total_checkups': 0,
            'latest_risk': None
        }

def get_user_activity_summary(user_id, cursor):
    """Get user's activity summary"""
    try:
        # Get recent activity data
        cursor.execute('''
            SELECT AVG(steps_per_day), AVG(calorie_intake), AVG(sleep_pattern)
            FROM CheckupData
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 5
        ''', (user_id,))
        avg_activity = cursor.fetchone()

        # Get last checkup date
        cursor.execute('''
            SELECT created_at
            FROM CheckupData
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (user_id,))
        last_checkup = cursor.fetchone()

        return {
            'avg_steps': int(avg_activity[0]) if avg_activity[0] else 0,
            'avg_calories': int(avg_activity[1]) if avg_activity[1] else 0,
            'avg_sleep': round(avg_activity[2], 1) if avg_activity[2] else 0,
            'last_checkup': last_checkup[0] if last_checkup else None
        }
    except:
        return {
            'avg_steps': 0,
            'avg_calories': 0,
            'avg_sleep': 0,
            'last_checkup': None
        }

def calculate_daily_calories(age, weight, height, gender='male', activity_level='moderate'):
    """Calculate daily calorie needs using Mifflin-St Jeor Equation"""
    # Base Metabolic Rate (BMR)
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Activity multipliers
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }

    return int(bmr * activity_multipliers.get(activity_level, 1.55))

def generate_diet_plan(age, height, weight, bmi, food_type, diseases, activity_level, health_data):
    """Generate a comprehensive personalized diet plan"""

    # Calculate daily calorie needs
    gender = 'male' if health_data.get('gender', 1) == 1 else 'female'
    daily_calories = calculate_daily_calories(age, weight, height, gender, activity_level)

    # Adjust calories based on BMI and health conditions
    if bmi < 18.5:  # Underweight
        daily_calories += 300
        weight_goal = "Gain weight gradually"
    elif bmi > 25:  # Overweight
        daily_calories -= 300
        weight_goal = "Lose weight gradually"
    else:
        weight_goal = "Maintain current weight"

    # Get meal plans based on food type and health conditions
    meal_plans = get_meal_plans(food_type, diseases, daily_calories)

    # Generate specific recommendations
    recommendations = get_health_recommendations(diseases, bmi, health_data)

    # Calculate macronutrient distribution
    macros = calculate_macros(daily_calories, diseases)

    # Generate shopping list
    shopping_list = generate_shopping_list(meal_plans)

    return {
        'daily_calories': daily_calories,
        'weight_goal': weight_goal,
        'bmi_category': get_bmi_category(bmi),
        'meal_plans': meal_plans,
        'recommendations': recommendations,
        'macros': macros,
        'shopping_list': shopping_list,
        'hydration': calculate_water_intake(weight),
        'exercise_tips': get_exercise_recommendations(bmi, diseases, activity_level)
    }

def get_bmi_category(bmi):
    """Get BMI category"""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_water_intake(weight):
    """Calculate daily water intake in liters"""
    return round(weight * 0.035, 1)

def calculate_macros(daily_calories, diseases):
    """Calculate macronutrient distribution based on health conditions"""
    # Default distribution
    carb_percent = 50
    protein_percent = 20
    fat_percent = 30

    # Adjust for specific conditions
    if 'diabetes' in diseases:
        carb_percent = 40  # Lower carbs for diabetes
        protein_percent = 25
        fat_percent = 35

    if 'heart_disease' in diseases or 'cholesterol' in diseases:
        fat_percent = 25  # Lower fat for heart health
        carb_percent = 55
        protein_percent = 20

    if 'kidney_disease' in diseases:
        protein_percent = 15  # Lower protein for kidney health
        carb_percent = 60
        fat_percent = 25

    return {
        'carbs': {
            'percent': carb_percent,
            'grams': int((daily_calories * carb_percent / 100) / 4),
            'calories': int(daily_calories * carb_percent / 100)
        },
        'protein': {
            'percent': protein_percent,
            'grams': int((daily_calories * protein_percent / 100) / 4),
            'calories': int(daily_calories * protein_percent / 100)
        },
        'fat': {
            'percent': fat_percent,
            'grams': int((daily_calories * fat_percent / 100) / 9),
            'calories': int(daily_calories * fat_percent / 100)
        }
    }

def get_meal_plans(food_type, diseases, daily_calories):
    """Generate meal plans based on dietary preferences and health conditions"""

    # Base meal structure
    meals = {
        'breakfast': {'calories': int(daily_calories * 0.25), 'foods': []},
        'lunch': {'calories': int(daily_calories * 0.35), 'foods': []},
        'dinner': {'calories': int(daily_calories * 0.30), 'foods': []},
        'snacks': {'calories': int(daily_calories * 0.10), 'foods': []}
    }

    # Food databases
    vegetarian_foods = {
        'breakfast': [
            'Oatmeal with berries and nuts (300 cal)',
            'Whole grain toast with avocado (250 cal)',
            'Greek yogurt with fruits (200 cal)',
            'Smoothie with spinach, banana, protein powder (280 cal)',
            'Quinoa porridge with almonds (320 cal)'
        ],
        'lunch': [
            'Quinoa salad with vegetables (400 cal)',
            'Lentil soup with whole grain bread (450 cal)',
            'Chickpea curry with brown rice (500 cal)',
            'Vegetable stir-fry with tofu (380 cal)',
            'Black bean burrito bowl (420 cal)'
        ],
        'dinner': [
            'Grilled vegetables with quinoa (350 cal)',
            'Lentil dal with roti (400 cal)',
            'Stuffed bell peppers with rice (380 cal)',
            'Vegetable pasta with olive oil (450 cal)',
            'Paneer curry with vegetables (420 cal)'
        ],
        'snacks': [
            'Mixed nuts and seeds (150 cal)',
            'Apple with almond butter (120 cal)',
            'Hummus with carrot sticks (100 cal)',
            'Greek yogurt (80 cal)',
            'Handful of berries (60 cal)'
        ]
    }

    non_vegetarian_foods = {
        'breakfast': [
            'Scrambled eggs with whole grain toast (320 cal)',
            'Greek yogurt with berries (200 cal)',
            'Omelette with vegetables (280 cal)',
            'Protein smoothie with banana (300 cal)',
            'Boiled eggs with avocado toast (350 cal)'
        ],
        'lunch': [
            'Grilled chicken salad (400 cal)',
            'Fish curry with brown rice (480 cal)',
            'Chicken soup with vegetables (350 cal)',
            'Tuna sandwich on whole grain (420 cal)',
            'Lean beef stir-fry (450 cal)'
        ],
        'dinner': [
            'Grilled salmon with vegetables (400 cal)',
            'Chicken breast with quinoa (380 cal)',
            'Fish with sweet potato (420 cal)',
            'Lean turkey with brown rice (400 cal)',
            'Grilled chicken with salad (350 cal)'
        ],
        'snacks': [
            'Boiled egg (70 cal)',
            'Chicken strips (100 cal)',
            'Tuna on crackers (120 cal)',
            'Protein bar (150 cal)',
            'Greek yogurt (80 cal)'
        ]
    }

    # Select appropriate food database
    if food_type == 'vegetarian':
        food_db = vegetarian_foods
    elif food_type == 'eggetarian':
        food_db = vegetarian_foods.copy()
        # Add egg options to eggetarian
        food_db['breakfast'].extend([
            'Scrambled eggs with toast (300 cal)',
            'Boiled eggs with fruit (250 cal)'
        ])
        food_db['snacks'].extend(['Boiled egg (70 cal)'])
    else:  # non-vegetarian
        food_db = non_vegetarian_foods

    # Populate meals
    import random
    for meal_type in meals:
        available_foods = food_db[meal_type].copy()

        # Adjust for health conditions
        if 'diabetes' in diseases:
            # Filter out high-carb options and add diabetes-friendly alternatives
            available_foods = [food for food in available_foods if 'rice' not in food.lower()]
            if meal_type == 'breakfast':
                available_foods.append('Steel-cut oats with cinnamon (250 cal)')
            elif meal_type == 'lunch':
                available_foods.append('Cauliflower rice bowl (300 cal)')

        if 'heart_disease' in diseases or 'cholesterol' in diseases:
            # Add heart-healthy options
            if meal_type == 'lunch':
                available_foods.append('Salmon salad with olive oil (350 cal)')
            elif meal_type == 'dinner':
                available_foods.append('Steamed fish with vegetables (320 cal)')

        # Select 2-3 options for variety
        selected_foods = random.sample(available_foods, min(3, len(available_foods)))
        meals[meal_type]['foods'] = selected_foods

    return meals

def get_health_recommendations(diseases, bmi, health_data):
    """Generate health-specific dietary recommendations"""
    recommendations = []

    # BMI-based recommendations
    if bmi < 18.5:
        recommendations.append("Focus on nutrient-dense, calorie-rich foods to gain healthy weight")
        recommendations.append("Include healthy fats like nuts, avocados, and olive oil")
    elif bmi > 25:
        recommendations.append("Focus on portion control and low-calorie, nutrient-dense foods")
        recommendations.append("Increase fiber intake with vegetables and whole grains")

    # Disease-specific recommendations
    if 'diabetes' in diseases:
        recommendations.extend([
            "Choose complex carbohydrates over simple sugars",
            "Monitor portion sizes and eat regular meals",
            "Include high-fiber foods to help control blood sugar",
            "Limit processed foods and sugary drinks"
        ])

    if 'hypertension' in diseases:
        recommendations.extend([
            "Reduce sodium intake to less than 2,300mg per day",
            "Increase potassium-rich foods like bananas and leafy greens",
            "Follow the DASH diet principles",
            "Limit alcohol consumption"
        ])

    if 'heart_disease' in diseases or 'cholesterol' in diseases:
        recommendations.extend([
            "Choose lean proteins and limit saturated fats",
            "Include omega-3 rich foods like fish and walnuts",
            "Increase soluble fiber intake",
            "Limit trans fats and processed foods"
        ])

    if 'kidney_disease' in diseases:
        recommendations.extend([
            "Limit protein intake as advised by your doctor",
            "Monitor phosphorus and potassium intake",
            "Control fluid intake if recommended",
            "Reduce sodium consumption"
        ])

    # General healthy eating tips
    recommendations.extend([
        "Stay hydrated with plenty of water",
        "Eat a variety of colorful fruits and vegetables",
        "Choose whole grains over refined grains",
        "Practice mindful eating and chew slowly"
    ])

    return recommendations

def get_exercise_recommendations(bmi, diseases, activity_level):
    """Generate exercise recommendations based on health status"""
    recommendations = []

    if bmi < 18.5:
        recommendations.extend([
            "Focus on strength training to build muscle mass",
            "Include resistance exercises 3-4 times per week",
            "Combine with moderate cardio for overall health"
        ])
    elif bmi > 25:
        recommendations.extend([
            "Start with low-impact cardio like walking or swimming",
            "Gradually increase intensity and duration",
            "Include strength training to preserve muscle mass",
            "Aim for 150 minutes of moderate exercise per week"
        ])
    else:
        recommendations.extend([
            "Maintain current activity level",
            "Mix cardio and strength training",
            "Try new activities to stay motivated"
        ])

    # Disease-specific exercise recommendations
    if 'diabetes' in diseases:
        recommendations.append("Exercise regularly to help control blood sugar levels")

    if 'heart_disease' in diseases:
        recommendations.append("Consult your doctor before starting any exercise program")

    if 'hypertension' in diseases:
        recommendations.append("Regular aerobic exercise can help lower blood pressure")

    return recommendations

def generate_shopping_list(meal_plans):
    """Generate a shopping list based on meal plans"""
    shopping_list = {
        'Proteins': [],
        'Vegetables': [],
        'Fruits': [],
        'Grains': [],
        'Dairy': [],
        'Pantry': []
    }

    # Extract ingredients from meal plans
    all_foods = []
    for meal_type in meal_plans:
        all_foods.extend(meal_plans[meal_type]['foods'])

    # Categorize common ingredients
    for food in all_foods:
        food_lower = food.lower()

        if any(protein in food_lower for protein in ['chicken', 'fish', 'salmon', 'tuna', 'eggs', 'tofu', 'lentil', 'chickpea', 'beans']):
            if 'chicken' in food_lower:
                shopping_list['Proteins'].append('Chicken breast')
            if 'fish' in food_lower or 'salmon' in food_lower:
                shopping_list['Proteins'].append('Fresh fish/salmon')
            if 'eggs' in food_lower:
                shopping_list['Proteins'].append('Eggs')
            if 'tofu' in food_lower:
                shopping_list['Proteins'].append('Tofu')
            if 'lentil' in food_lower:
                shopping_list['Proteins'].append('Lentils')

        if any(veg in food_lower for veg in ['vegetables', 'spinach', 'carrot', 'bell pepper', 'broccoli']):
            shopping_list['Vegetables'].extend(['Mixed vegetables', 'Leafy greens', 'Bell peppers'])

        if any(fruit in food_lower for fruit in ['berries', 'banana', 'apple', 'avocado']):
            shopping_list['Fruits'].extend(['Mixed berries', 'Bananas', 'Apples', 'Avocados'])

        if any(grain in food_lower for grain in ['quinoa', 'rice', 'oats', 'bread', 'pasta']):
            shopping_list['Grains'].extend(['Quinoa', 'Brown rice', 'Oats', 'Whole grain bread'])

        if any(dairy in food_lower for dairy in ['yogurt', 'milk', 'cheese']):
            shopping_list['Dairy'].extend(['Greek yogurt', 'Low-fat milk'])

    # Remove duplicates and add pantry staples
    for category in shopping_list:
        shopping_list[category] = list(set(shopping_list[category]))

    shopping_list['Pantry'] = [
        'Olive oil', 'Nuts and seeds', 'Spices and herbs',
        'Whole grain cereals', 'Healthy snacks'
    ]

    return shopping_list

def generate_enhanced_charts(df):
    """Generate enhanced charts with better styling and insights"""
    charts = {}

    # Define chart categories and their metrics
    chart_categories = {
        'vital_signs': {
            'title': 'Vital Signs',
            'metrics': ['systole', 'diastole', 'heart_rate'],
            'color': '#FF6B6B'
        },
        'blood_metrics': {
            'title': 'Blood Metrics',
            'metrics': ['rbc_count', 'wbc_count', 'blood_sugar', 'cholesterol'],
            'color': '#4ECDC4'
        },
        'lifestyle': {
            'title': 'Lifestyle Metrics',
            'metrics': ['bmi', 'steps_per_day', 'calorie_intake'],
            'color': '#45B7D1'
        }
    }

    for category, info in chart_categories.items():
        for metric in info['metrics']:
            if metric in df.columns:
                # Create enhanced line chart
                fig = px.line(df, x='created_at', y=metric,
                            title=f'{metric.replace("_", " ").title()} Trend',
                            color_discrete_sequence=[info['color']])

                # Enhance chart styling
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial, sans-serif", size=12),
                    title=dict(font=dict(size=16, color='#2C3E50')),
                    xaxis=dict(
                        title='Date',
                        gridcolor='rgba(128,128,128,0.2)',
                        showgrid=True
                    ),
                    yaxis=dict(
                        title=metric.replace('_', ' ').title(),
                        gridcolor='rgba(128,128,128,0.2)',
                        showgrid=True
                    ),
                    hovermode='x unified'
                )

                # Add trend line if enough data points
                if len(df) > 2:
                    fig.add_scatter(x=df['created_at'], y=df[metric].rolling(window=2).mean(),
                                  mode='lines', name='Trend',
                                  line=dict(dash='dash', color='rgba(255,0,0,0.5)'))

                charts[metric] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return charts

def calculate_summary_statistics(df):
    """Calculate comprehensive summary statistics"""
    stats = {}

    # Key health metrics
    key_metrics = {
        'bmi': {'name': 'BMI', 'unit': '', 'normal_range': (18.5, 25)},
        'systole': {'name': 'Systolic BP', 'unit': 'mmHg', 'normal_range': (90, 120)},
        'diastole': {'name': 'Diastolic BP', 'unit': 'mmHg', 'normal_range': (60, 80)},
        'heart_rate': {'name': 'Heart Rate', 'unit': 'bpm', 'normal_range': (60, 100)},
        'blood_sugar': {'name': 'Blood Sugar', 'unit': 'mg/dL', 'normal_range': (70, 100)},
        'cholesterol': {'name': 'Cholesterol', 'unit': 'mg/dL', 'normal_range': (0, 200)},
        'steps_per_day': {'name': 'Daily Steps', 'unit': 'steps', 'normal_range': (8000, 12000)},
        'calorie_intake': {'name': 'Daily Calories', 'unit': 'cal', 'normal_range': (1800, 2500)}
    }

    for metric, info in key_metrics.items():
        if metric in df.columns and not df[metric].empty:
            current_value = df[metric].iloc[-1]  # Latest value
            avg_value = df[metric].mean()
            min_value = df[metric].min()
            max_value = df[metric].max()

            # Determine status based on normal range
            normal_min, normal_max = info['normal_range']
            if normal_min <= current_value <= normal_max:
                status = 'Normal'
                status_color = '#28a745'
            elif current_value < normal_min:
                status = 'Below Normal'
                status_color = '#ffc107'
            else:
                status = 'Above Normal'
                status_color = '#dc3545'

            stats[metric] = {
                'name': info['name'],
                'current': round(current_value, 1),
                'average': round(avg_value, 1),
                'min': round(min_value, 1),
                'max': round(max_value, 1),
                'unit': info['unit'],
                'status': status,
                'status_color': status_color,
                'normal_range': f"{normal_min}-{normal_max} {info['unit']}"
            }

    return stats

def analyze_health_trends(df):
    """Analyze health trends and patterns"""
    trends = {}

    if len(df) < 2:
        return trends

    # Analyze trends for key metrics
    trend_metrics = ['bmi', 'systole', 'diastole', 'heart_rate', 'blood_sugar', 'cholesterol']

    for metric in trend_metrics:
        if metric in df.columns and not df[metric].empty:
            # Calculate trend direction
            recent_avg = df[metric].tail(3).mean() if len(df) >= 3 else df[metric].iloc[-1]
            older_avg = df[metric].head(3).mean() if len(df) >= 3 else df[metric].iloc[0]

            change = recent_avg - older_avg
            change_percent = (change / older_avg * 100) if older_avg != 0 else 0

            if abs(change_percent) < 2:
                direction = 'Stable'
                icon = '→'
                color = '#6c757d'
            elif change_percent > 0:
                direction = 'Increasing'
                icon = '↗'
                color = '#dc3545' if metric in ['systole', 'diastole', 'blood_sugar', 'cholesterol'] else '#28a745'
            else:
                direction = 'Decreasing'
                icon = '↘'
                color = '#28a745' if metric in ['systole', 'diastole', 'blood_sugar', 'cholesterol'] else '#dc3545'

            trends[metric] = {
                'direction': direction,
                'change_percent': abs(round(change_percent, 1)),
                'icon': icon,
                'color': color,
                'change_value': round(abs(change), 1)
            }

    return trends

def generate_health_recommendations_from_data(df, risk_data):
    """Generate personalized health recommendations based on user data"""
    recommendations = []

    if df.empty:
        return recommendations

    # Get latest values
    latest = df.iloc[-1]

    # BMI recommendations
    if 'bmi' in df.columns:
        bmi = latest['bmi']
        if bmi < 18.5:
            recommendations.append({
                'category': 'Weight Management',
                'message': 'Your BMI indicates you are underweight. Consider consulting a nutritionist for a healthy weight gain plan.',
                'priority': 'medium'
            })
        elif bmi > 25:
            recommendations.append({
                'category': 'Weight Management',
                'message': 'Your BMI indicates you are overweight. Focus on a balanced diet and regular exercise.',
                'priority': 'high'
            })

    # Blood pressure recommendations
    if 'systole' in df.columns and 'diastole' in df.columns:
        systole = latest['systole']
        diastole = latest['diastole']
        if systole > 140 or diastole > 90:
            recommendations.append({
                'category': 'Blood Pressure',
                'message': 'Your blood pressure is elevated. Consider reducing sodium intake and increasing physical activity.',
                'priority': 'high'
            })

    # Blood sugar recommendations
    if 'blood_sugar' in df.columns:
        blood_sugar = latest['blood_sugar']
        if blood_sugar > 100:
            recommendations.append({
                'category': 'Blood Sugar',
                'message': 'Your blood sugar levels are elevated. Monitor your carbohydrate intake and consider regular exercise.',
                'priority': 'high'
            })

    # Activity recommendations
    if 'steps_per_day' in df.columns:
        steps = latest['steps_per_day']
        if steps < 8000:
            recommendations.append({
                'category': 'Physical Activity',
                'message': 'Try to increase your daily steps to at least 8,000-10,000 for better cardiovascular health.',
                'priority': 'medium'
            })

    # Cholesterol recommendations
    if 'cholesterol' in df.columns:
        cholesterol = latest['cholesterol']
        if cholesterol > 200:
            recommendations.append({
                'category': 'Cholesterol',
                'message': 'Your cholesterol levels are high. Consider a heart-healthy diet rich in omega-3 fatty acids.',
                'priority': 'high'
            })

    # Add general recommendations
    recommendations.extend([
        {
            'category': 'General Health',
            'message': 'Maintain regular checkups to monitor your health trends and catch any issues early.',
            'priority': 'low'
        },
        {
            'category': 'Lifestyle',
            'message': 'Stay hydrated, get adequate sleep (7-9 hours), and manage stress through relaxation techniques.',
            'priority': 'medium'
        }
    ])

    return recommendations



@app.route('/analysis')
def analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    user_name = session.get('name', 'User')

    # Connect to the database and retrieve the user's checkup data
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT created_at, rbc_count, wbc_count, systole, diastole, heart_rate, allergies, blood_sugar, bmi, cholesterol, steps_per_day, calorie_intake
        FROM CheckupData WHERE user_id = ? ORDER BY created_at
    ''', (user_id,))
    checkup_data = cursor.fetchall()

    # Get risk data for additional insights
    cursor.execute('''
        SELECT r.created_at, r.stress_level, r.risk_score, r.diseases
        FROM ResultData r
        JOIN CheckupData c ON r.report_id = c.checkup_id
        WHERE c.user_id = ?
        ORDER BY r.created_at
    ''', (user_id,))
    risk_data = cursor.fetchall()
    conn.close()

    # Check if user has any data
    if not checkup_data:
        return render_template('analysis.html',
                             charts={},
                             summary_stats={},
                             trends={},
                             recommendations=[],
                             user_name=user_name,
                             no_data=True)

    # Convert the data into a DataFrame for easier manipulation with Plotly
    df = pd.DataFrame(checkup_data, columns=['created_at', 'rbc_count', 'wbc_count', 'systole', 'diastole', 'heart_rate', 'allergies', 'blood_sugar', 'bmi', 'cholesterol', 'steps_per_day', 'calorie_intake'])

    # Convert 'created_at' to datetime format
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Generate enhanced charts with better styling
    charts = generate_enhanced_charts(df)

    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(df)

    # Analyze trends
    trends = analyze_health_trends(df)

    # Generate health recommendations based on data
    recommendations = generate_health_recommendations_from_data(df, risk_data)

    return render_template('analysis.html',
                         charts=charts,
                         summary_stats=summary_stats,
                         trends=trends,
                         recommendations=recommendations,
                         user_name=user_name,
                         no_data=False)


@app.route('/result')
def result():
    user_name=session.get('name', 'User')
    return render_template('result.html',name=user_name)


@app.route('/instant_checkup', methods=['GET', 'POST'])
def instant_checkup():
    user_id = session.get('user_id')

    # Default values for the form
    default_data = [25, 1, 4.5, 7000, 120, 80, 75, 0, 95, 22.5, 180, 8000, 2000, 0, 0, 8, 0, 'P01']

    # Try to get the user's most recent checkup data for pre-filling
    if user_id:
        try:
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT age, gender, rbc_count, wbc_count, systole, diastole, heart_rate,
                       allergies, blood_sugar, bmi, cholesterol, steps_per_day, calorie_intake,
                       smoking, alcohol, sleep_pattern, num_claims, procedure_codes
                FROM CheckupData
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (user_id,))
            recent_data = cursor.fetchone()
            conn.close()

            if recent_data:
                checkup_data = list(recent_data)
            else:
                checkup_data = default_data
        except:
            checkup_data = default_data
    else:
        checkup_data = default_data

    return render_template('instant_checkup.html', checkup_data=checkup_data)
# ---------------- Doctor Auth Helper + Decorator ---------------- #
# ================= Doctor Auth Helper + Decorator =================




def check_doctor(email, password):
    """
    Check doctor credentials against Doctors table (plain text password).
    Table schema:
      Doctors(doctor_id INTEGER PK, name TEXT, email TEXT UNIQUE, password TEXT, created_at TEXT)
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT doctor_id, name, email, password FROM Doctors WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row and row["password"] == password:  # plain-text check (per your request)
        return {"doctor_id": row["doctor_id"], "name": row["name"], "email": row["email"]}
    return None


def doctor_login_required(view_func):
    """Decorator to restrict routes to logged-in doctors only."""
    from functools import wraps
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("doctor_id"):
            flash("Doctor login required.", "warning")
            return redirect(url_for("doctor_login"))
        return view_func(*args, **kwargs)
    return wrapper


# ======================== Doctor Routes ===========================
@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        doc = check_doctor(email, password)
        if doc:
            session["doctor_id"] = doc["doctor_id"]
            session["doctor_name"] = doc["name"]
            flash(f"Welcome, Dr. {doc['name']}", "success")
            return redirect(url_for("doctor_home"))
        flash("Invalid doctor credentials.", "danger")
    return render_template("doctor_login.html")


@app.route("/doctor/logout")
def doctor_logout():
    session.pop("doctor_id", None)
    session.pop("doctor_name", None)
    flash("Signed out.", "info")
    return redirect(url_for("index"))


@app.route("/doctor/home")
@doctor_login_required
def doctor_home():
    return render_template("doctor_home.html", doctor_name=session.get("doctor_name", "Doctor"))


@app.route("/doctor/reports")
@doctor_login_required
def doctor_reports():
    """
    All patient checkups with their latest (or all) results.
    Join per your schema: Users.id ↔ CheckupData.user_id and ResultData.user_id ↔ Users.id
    Cast numeric fields so Jinja won't try to round strings.
    """
    q = request.args.get("q", "").strip()  # optional search by name/email
    conn = get_db()
    cur = conn.cursor()
    sql = """
        SELECT
            c.created_at                      AS checkup_time,     -- [0]
            c.checkup_id                      AS checkup_id,       -- [1]
            u.id                              AS user_id,          -- [2]
            u.name                            AS patient_name,     -- [3]
            COALESCE(u.email,'')              AS patient_email,    -- [4]
            c.age,                            -- [5]
            c.gender,                         -- [6]
            c.bmi,                            -- [7]
            c.systole,                        -- [8]
            c.diastole,                       -- [9]
            c.heart_rate,                     -- [10]
            r.stress_level,                   -- [11]
            r.diseases,                       -- [12]
            CAST(r.risk_score AS REAL)        AS risk_score,       -- [13]
            CAST(r.predicted_los AS REAL)     AS predicted_los,    -- [14]
            r.readmission_label,              -- [15]
            CAST(r.readmission_prob AS REAL)  AS readmission_prob, -- [16]
            r.created_at                      AS result_time       -- [17]
        FROM CheckupData c
        JOIN Users u         ON u.id = c.user_id
        LEFT JOIN ResultData r ON r.user_id = u.id
        WHERE 1=1
    """
    params = []
    if q:
        sql += " AND (u.name LIKE ? OR u.email LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%"])
    sql += " ORDER BY c.created_at DESC, r.created_at DESC"

    cur.execute(sql, params)
    reports = cur.fetchall()
    conn.close()

    return render_template("doctor_reports.html", reports=reports, q=q)



@app.route('/system_analysis')
def system_analysis():
    """Analyze current readmission prediction system performance"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        # Load and analyze current data
        analysis_results = analyze_current_system()
        return render_template('system_analysis.html', analysis=analysis_results)
    except Exception as e:
        flash(f'Error analyzing system: {str(e)}', 'error')
        return redirect(url_for('home'))

def analyze_current_system():
    """Comprehensive analysis of current readmission prediction system"""
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score

    # Load the merged dataset
    df = pd.read_csv('data/merged.csv')

    # Load claim data with readmission outcomes
    claim_df = pd.read_csv('newdatasets/claim_new.csv')

    analysis = {
        'data_quality': {},
        'model_performance': {},
        'feature_analysis': {},
        'recommendations': []
    }

    # Data Quality Analysis
    analysis['data_quality'] = {
        'total_records': len(df),
        'total_claims': len(claim_df),
        'missing_values': df.isnull().sum().to_dict(),
        'readmission_rate': claim_df['readmission_30d'].mean() if 'readmission_30d' in claim_df.columns else 0,
        'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0
    }

    # Current Model Performance Analysis
    if readm_model and 'readmission_30d' in claim_df.columns:
        try:
            # Prepare features for current model
            X = claim_df[CLAIM_FEATURES].copy()
            y = claim_df['readmission_30d']

            # Handle categorical variables
            from sklearn.preprocessing import LabelEncoder
            le_service = LabelEncoder()
            le_diag = LabelEncoder()
            le_proc = LabelEncoder()

            X['service_type'] = le_service.fit_transform(X['service_type'].astype(str))
            X['diagnosis_code'] = le_diag.fit_transform(X['diagnosis_code'].astype(str))
            X['procedure_code'] = le_proc.fit_transform(X['procedure_code'].astype(str))

            # Get predictions
            y_pred = readm_model.predict(X)
            y_pred_proba = readm_model.predict_proba(X)[:, 1]

            analysis['model_performance'] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_pred_proba),
                'cross_val_accuracy': cross_val_score(readm_model, X, y, cv=5).mean()
            }
        except Exception as e:
            analysis['model_performance'] = {'error': str(e)}

    # Feature Analysis
    analysis['feature_analysis'] = {
        'current_features': CLAIM_FEATURES,
        'available_clinical_features': [col for col in df.columns if col in DISEASE_FEATURES],
        'missing_important_features': [
            'comorbidity_count', 'previous_admissions', 'social_determinants',
            'medication_adherence', 'discharge_disposition', 'insurance_type'
        ]
    }

    # Recommendations for improvement
    current_accuracy = analysis['model_performance'].get('accuracy', 0)
    if current_accuracy < 0.8:
        analysis['recommendations'].extend([
            f"Current accuracy ({current_accuracy:.2%}) is below 80% target",
            "Incorporate additional clinical features (comorbidities, vital signs)",
            "Add social determinants of health data",
            "Include historical admission patterns",
            "Implement ensemble modeling approaches"
        ])

    analysis['recommendations'].extend([
        "Implement real-time data validation",
        "Add feature engineering for interaction terms",
        "Develop risk stratification thresholds",
        "Create intervention recommendation engine"
    ])

    return analysis




# ================= Hospital Auth Helper + Decorator =================
def check_hospital(email, password):
    """Plain-text credential check against Hospitals table."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT hospital_id, name, email, password FROM Hospitals WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if row and row["password"] == password:
        return {"hospital_id": row["hospital_id"], "name": row["name"], "email": row["email"]}
    return None

def hospital_login_required(view_func):
    from functools import wraps
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("hospital_id"):
            flash("Hospital login required.", "warning")
            return redirect(url_for("hospital_login"))
        return view_func(*args, **kwargs)
    return wrapper











# ========================= Hospital Routes =========================
@app.route("/hospital/login", methods=["GET", "POST"])
def hospital_login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        hosp = check_hospital(email, password)
        if hosp:
            session["hospital_id"] = hosp["hospital_id"]
            session["hospital_name"] = hosp["name"]
            flash(f"Welcome, {hosp['name']}", "success")
            return redirect(url_for("hospital_home"))
        flash("Invalid hospital credentials.", "danger")
    return render_template("hospital_login.html")

@app.route("/hospital/logout")
def hospital_logout():
    session.pop("hospital_id", None)
    session.pop("hospital_name", None)
    flash("Signed out.", "info")
    return redirect(url_for("index"))

@app.route("/hospital/home")
@hospital_login_required
def hospital_home():
    hospital_name = session.get("hospital_name", "Hospital")
    hospital_id = session.get("hospital_id")  # should be an int
    try:
        hospital_id = int(hospital_id) if hospital_id is not None else None
    except:
        hospital_id = None

    print("DEBUG /hospital/home: session.hospital_id =", hospital_id)

    conn = get_db()
    cur = conn.cursor()

    # Primary query: costs for the logged-in hospital
    cur.execute("""
        SELECT month, total_cost
        FROM HospitalCosts
        WHERE hospital_id = ?
        ORDER BY month DESC
        LIMIT 12
    """, (hospital_id,))
    rows = cur.fetchall()

    fallback_notice = None

    # Fallback: if no rows found for this hospital, show any data to prove DB is connected
    if not rows:
        fallback_notice = ("No HospitalCosts rows found for your hospital_id "
                           f"({hospital_id}). Showing all hospitals (fallback) below.")
        cur.execute("""
            SELECT month, total_cost
            FROM HospitalCosts
            ORDER BY month DESC
            LIMIT 12
        """)
        rows = cur.fetchall()

    conn.close()

    # Extract lists
    months = [r[0] for r in rows]
    costs  = [float(r[1]) for r in rows]

    # Compute comparison with previous month
    current_month = months[0] if months else None
    previous_month = months[1] if len(months) > 1 else None
    current_cost  = costs[0] if costs else None
    prev_cost     = costs[1] if len(costs) > 1 else None

    delta = None
    delta_pct = None
    trend = "flat"
    if current_cost is not None and prev_cost is not None:
        delta = current_cost - prev_cost
        delta_pct = (delta / prev_cost * 100.0) if prev_cost != 0 else None
        trend = "down" if delta < 0 else ("up" if delta > 0 else "flat")

    # Ascending order for the table
    data = list(zip(months[::-1], costs[::-1]))

    return render_template(
        "hospital_home.html",
        hospital_name=hospital_name,
        current_month=current_month,
        previous_month=previous_month,
        current_cost=current_cost,
        prev_cost=prev_cost,
        delta=delta,
        delta_pct=delta_pct,
        trend=trend,
        data=data,
        fallback_notice=fallback_notice
    )




@app.route("/hospital/costs", methods=["GET", "POST"])
@hospital_login_required
def hospital_costs():
    """View/add monthly costs."""
    hospital_id = session["hospital_id"]
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        month = request.form.get("month", "").strip()        # 'YYYY-MM'
        total_cost = float(request.form.get("total_cost", 0) or 0)
        notes = request.form.get("notes", "")
        try:
            cur.execute("""
                INSERT OR REPLACE INTO HospitalCosts (hospital_id, month, total_cost, notes)
                VALUES (?, ?, ?, ?)
            """, (hospital_id, month, total_cost, notes))
            conn.commit()
            flash("Monthly cost saved.", "success")
        except Exception as e:
            flash("Error saving cost.", "danger")

    cur.execute("""
        SELECT month, total_cost, notes, created_at
        FROM HospitalCosts
        WHERE hospital_id = ?
        ORDER BY month DESC
    """, (hospital_id,))
    all_costs = cur.fetchall()
    conn.close()

    return render_template("hospital_costs.html", all_costs=all_costs)

# ---------------------- EHR: Hospital side -----------------------
@app.route("/hospital/ehr", methods=["GET", "POST"])
@hospital_login_required
def hospital_ehr():
    """Create/list EHR requests."""
    hospital_id = session["hospital_id"]
    conn = get_db()
    cur = conn.cursor()

    if request.method == "POST":
        request_type = request.form.get("request_type", "").strip()
        details = request.form.get("details", "")
        doctor_id = request.form.get("doctor_id")  # optional
        doctor_id = int(doctor_id) if doctor_id else None
        patient_user_id = request.form.get("patient_user_id")
        patient_user_id = int(patient_user_id) if patient_user_id else None

        cur.execute("""
            INSERT INTO EHRRequests (hospital_id, patient_user_id, doctor_id, request_type, details, status)
            VALUES (?, ?, ?, ?, ?, 'PENDING')
        """, (hospital_id, patient_user_id, doctor_id, request_type, details))
        conn.commit()
        flash("EHR request created.", "success")

    # List requests for this hospital
    cur.execute("""
        SELECT e.ehr_id, e.created_at, e.request_type, e.details, e.status,
               e.patient_user_id, e.doctor_id, e.decided_at, e.decided_by_doctor_id
        FROM EHRRequests e
        WHERE e.hospital_id = ?
        ORDER BY e.created_at DESC
    """, (hospital_id,))
    reqs = cur.fetchall()
    conn.close()

    return render_template("hospital_ehr.html", reqs=reqs)











# ---------------------- EHR: Doctor side ------------------------
@app.route("/doctor/ehr")
@doctor_login_required
def doctor_ehr_list():
    """List all EHR requests assigned to the logged-in doctor (or all if you prefer)."""
    doctor_id = session.get("doctor_id")
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT e.ehr_id, e.created_at, h.name AS hospital_name, e.request_type, e.details,
               e.status, e.patient_user_id, e.doctor_id, e.decided_at
        FROM EHRRequests e
        JOIN Hospitals h ON h.hospital_id = e.hospital_id
        WHERE e.doctor_id = ?
        ORDER BY e.created_at DESC
    """, (doctor_id,))
    rows = cur.fetchall()
    conn.close()

    return render_template("doctor_ehr.html", rows=rows)

@app.route("/doctor/ehr/<int:ehr_id>/action", methods=["POST"])
@doctor_login_required
def doctor_ehr_action(ehr_id):
    """Approve / Reject an EHR request."""
    action = request.form.get("action")  # 'APPROVE' or 'REJECT'
    doctor_id = session.get("doctor_id")

    new_status = "APPROVED" if action == "APPROVE" else "REJECTED"
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        UPDATE EHRRequests
        SET status = ?, decided_at = datetime('now','localtime'), decided_by_doctor_id = ?
        WHERE ehr_id = ?
    """, (new_status, doctor_id, ehr_id))
    conn.commit()
    conn.close()
    flash(f"Request {new_status.lower()}.", "success")
    return redirect(url_for("doctor_ehr_list"))



















@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)








if __name__ == '__main__':
    app.run(debug=True)
    

