import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load model
MODEL_PATH = 'Gradient.pkl'
try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Attractive layout HTML & CSS Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --card-bg: rgba(15, 23, 42, 0.75);
            --input-bg: rgba(30, 41, 59, 0.6);
            --border: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Animated Vibrant Gradient Background */
        body {
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 15px;
            color: var(--text-main);
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #0284c7);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Glassmorphism Card */
        .container {
            width: 100%;
            max-width: 680px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 30px var(--accent-glow);
        }

        header {
            text-align: center;
            margin-bottom: 32px;
        }

        header h1 {
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #c084fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        header p {
            color: var(--text-sub);
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-sub);
            letter-spacing: 0.3px;
        }

        input, select {
            background-color: var(--input-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px 16px;
            color: var(--text-main);
            font-size: 14px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        input:focus, select:focus {
            outline: none;
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.2);
            background-color: rgba(30, 41, 59, 0.9);
        }

        /* Gradient Glowing Button */
        button {
            margin-top: 10px;
            grid-column: span 2;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.5);
            transition: all 0.3s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(124, 58, 237, 0.6);
        }

        button:active {
            transform: translateY(0);
        }

        /* Result Box with Animation */
        .result-card {
            margin-top: 28px;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            animation: slideUp 0.4s ease-out;
        }

        .result-card h3 {
            font-size: 13px;
            color: #6ee7b7;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
        }

        .result-card .prediction-value {
            font-size: 32px;
            font-weight: 700;
            color: #34d399;
            text-shadow: 0 0 12px rgba(52, 211, 153, 0.3);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, button {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Predictive AI Portal</h1>
            <p>Gradient Boosting Regressor Model</p>
        </header>

        <form method="POST">
            <div class="form-grid">
                <div class="form-group full-width">
                    <label for="Name">Candidate Name</label>
                    <input type="text" id="Name" name="Name" placeholder="Enter full name" required>
                </div>

                <div class="form-group">
                    <label for="Department">Department</label>
                    <select id="Department" name="Department" required>
                        <option value="" disabled selected>Select Department</option>
                        <option value="Engineering">Engineering</option>
                        <option value="Data Science">Data Science</option>
                        <option value="Product">Product</option>
                        <option value="Sales">Sales & Marketing</option>
                        <option value="HR">HR</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Experience_Years">Experience (Years)</label>
                    <input type="number" step="0.1" id="Experience_Years" name="Experience_Years" placeholder="e.g. 4.5" required>
                </div>

                <div class="form-group">
                    <label for="Education_Level">Education Level</label>
                    <select id="Education_Level" name="Education_Level" required>
                        <option value="" disabled selected>Select Education</option>
                        <option value="Bachelor">Bachelor's</option>
                        <option value="Master">Master's</option>
                        <option value="PhD">PhD</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="Age">Age</label>
                    <input type="number" id="Age" name="Age" placeholder="e.g. 26" required>
                </div>

                <div class="form-group">
                    <label for="Gender">Gender</label>
                    <select id="Gender" name="Gender" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="City">City</label>
                    <input type="text" id="City" name="City" placeholder="e.g. Pune" required>
                </div>

                <button type="submit">Predict Outcome</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Estimated Prediction Output</h3>
            <div class="prediction-value">{{ prediction }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            name = request.form.get('Name')
            department = request.form.get('Department')
            exp_years = float(request.form.get('Experience_Years'))
            edu_level = request.form.get('Education_Level')
            age = float(request.form.get('Age'))
            gender = request.form.get('Gender')
            city = request.form.get('City')

            # Numeric Encodings for Categorical Features
            name_encoded = len(str(name))
            
            dept_map = {"Engineering": 0, "Data Science": 1, "Product": 2, "Sales": 3, "HR": 4}
            dept_encoded = dept_map.get(department, 0)

            edu_map = {"Bachelor": 0, "Master": 1, "PhD": 2, "Other": 3}
            edu_encoded = edu_map.get(edu_level, 0)

            gender_map = {"Male": 0, "Female": 1, "Other": 2}
            gender_encoded = gender_map.get(gender, 0)

            city_encoded = sum(ord(c) for c in str(city)) % 100

            input_data = pd.DataFrame([[
                name_encoded, dept_encoded, exp_years, edu_encoded, age, gender_encoded, city_encoded
            ]], columns=['Name', 'Department', 'Experience_Years', 'Education_Level', 'Age', 'Gender', 'City'])

            if model:
                pred_value = model.predict(input_data)[0]
                prediction = round(float(pred_value), 2)
            else:
                prediction = "Model file not loading."

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)