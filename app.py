# heart_disease_compare_models.py

# 1️⃣ Import libraries
from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

app = Flask(__name__)

# Map dataset names to files and targets
DATASETS = {
    "Heart Disease": {
        "file": "heart.csv",
        "target": "target"
    },
    "Diabetes": {
        "file": "diabetes.csv",
        "target": "Outcome"
    }
}

@app.route("/", methods=["GET", "POST"])
def index():
    selected_dataset = request.form.get("dataset", "Heart Disease")
    dataset_info = DATASETS[selected_dataset]
    data = pd.read_csv(dataset_info["file"])
    target_col = dataset_info["target"]

    X = data.drop(target_col, axis=1)
    y = data[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {"Accuracy": acc, "Confusion Matrix": cm, "Classification Report": report}

    # Clinical Trial Twin Simulator using CSV
    twin_file = "clinical_trial_twin.csv"
    try:
        twin_data = pd.read_csv(twin_file)
        features = [col for col in twin_data.columns if col != "treatment"]
        X_patients = scaler.transform(twin_data[features])
        rf_model = models["Random Forest"]
        twin_data["predicted_outcome"] = rf_model.predict(X_patients)
        drug_group = twin_data[twin_data["treatment"] == "drug"]
        placebo_group = twin_data[twin_data["treatment"] == "placebo"]
        drug_rate = drug_group["predicted_outcome"].mean()
        placebo_rate = placebo_group["predicted_outcome"].mean()
    except Exception as e:
        twin_data = pd.DataFrame()
        drug_rate = placebo_rate = None

    # HTML template
    html = """
    <html>
    <head>
        <title>Clinical Trial Digital Twin Simulator Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            table, th, td { border: 1px solid #aaa; border-collapse: collapse; padding: 6px; }
            th { background: #eee; }
        </style>
    </head>
    <body>
        <form method="post">
            <label for="dataset">Select Dataset:</label>
            <select name="dataset" id="dataset" onchange="this.form.submit()">
                {% for name in datasets %}
                    <option value="{{ name }}" {% if name == selected_dataset %}selected{% endif %}>{{ name }}</option>
                {% endfor %}
            </select>
        </form>
        <h1>Model Comparison Results ({{ selected_dataset }})</h1>
        {% for name, metrics in results.items() %}
            <h2>{{ name }}</h2>
            <b>Accuracy:</b> {{ "%.2f"|format(metrics['Accuracy']*100) }}%<br>
            <b>Confusion Matrix:</b>
            <table>
                {% for row in metrics['Confusion Matrix'] %}
                    <tr>
                    {% for val in row %}
                        <td>{{ val }}</td>
                    {% endfor %}
                    </tr>
                {% endfor %}
            </table>
            <b>Classification Report:</b>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-score</th>
                    <th>Support</th>
                </tr>
                {% for label, scores in metrics['Classification Report'].items() if label in ['0', '1'] %}
                <tr>
                    <td>{{ label }}</td>
                    <td>{{ "%.2f"|format(scores['precision']) }}</td>
                    <td>{{ "%.2f"|format(scores['recall']) }}</td>
                    <td>{{ "%.2f"|format(scores['f1-score']) }}</td>
                    <td>{{ scores['support'] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endfor %}

        <h1>Clinical Trial Digital Twin Simulator (CSV)</h1>
        {% if drug_rate is not none and placebo_rate is not none %}
            <b>Drug group outcome rate:</b> {{ "%.2f"|format(drug_rate*100) }}%<br>
            <b>Placebo group outcome rate:</b> {{ "%.2f"|format(placebo_rate*100) }}%<br>
        {% else %}
            <b>No clinical_trial_twin.csv found or columns mismatch.</b><br>
        {% endif %}

        {% if not twin_data.empty %}
        <h2>First 5 rows of Clinical Trial Twin CSV</h2>
        <table>
        <tr>
        {% for col in twin_data.columns %}
            <th>{{ col }}</th>
        {% endfor %}
        </tr>
        {% for row in twin_data.head(5).values %}
        <tr>
            {% for val in row %}
            <td>{{ val }}</td>
            {% endfor %}
        </tr>
        {% endfor %}
        </table>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(
        html,
        results=results,
        drug_rate=drug_rate,
        placebo_rate=placebo_rate,
        twin_data=twin_data,
        datasets=DATASETS.keys(),
        selected_dataset=selected_dataset
    )

@app.route("/drug_test", methods=["GET", "POST"])
def drug_test():
    message = ""
    results = None
    if request.method == "POST":
        file = request.files.get("patient_csv")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join("uploaded_patients.csv")
            file.save(filepath)
            patient_data = pd.read_csv(filepath)
            patient_data["treatment"] = "drug"
            data = pd.read_csv("heart.csv")
            X = data.drop("target", axis=1)
            y = data["target"]
            scaler = StandardScaler()
            scaler.fit(X)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(scaler.transform(X), y)
            # Check columns
            expected_cols = set(X.columns)
            patient_cols = set(patient_data.columns) - {"treatment"}
            if expected_cols != patient_cols:
                message = f"Column mismatch!<br>Expected: {sorted(expected_cols)}<br>Got: {sorted(patient_cols)}"
                results = None
            else:
                X_patients = scaler.transform(patient_data.drop("treatment", axis=1))
                patient_data["predicted_outcome"] = rf_model.predict(X_patients)
                results = patient_data[["predicted_outcome"]].value_counts().to_dict()
                message = "Prediction complete!"
        else:
            message = "Please upload a valid CSV file."
    html = """
    <html>
    <head>
        <title>Drug Reaction Test</title>
    </head>
    <body>
        <h1>Test Patient Reaction to Drug</h1>
        <form method="post" enctype="multipart/form-data">
            <label>Upload patient CSV:</label>
            <input type="file" name="patient_csv" accept=".csv" required>
            <button type="submit">Test Drug</button>
        </form>
        <p>{{ message|safe }}</p>
        {% if results %}
            <h2>Predicted Outcomes</h2>
            <ul>
            {% for outcome, count in results.items() %}
                <li>Outcome {{ outcome[0] }}: {{ count }}</li>
            {% endfor %}
            </ul>
        {% endif %}
    </body>
    </html>
    """
    return render_template_string(html, message=message, results=results)

if __name__ == "__main__":
    app.run(debug=True)

# Install required packages
# pip install flask pandas numpy scikit-learn

# import pandas as pd
# print(pd.read_csv("heart.csv").columns.tolist())
# print(pd.read_csv("uploaded_patients.csv").columns.tolist())
if __name__ == "__main__":
    app.run(debug=True)

import pandas as pd

df = pd.read_csv("stroke_data.csv")
df = df.drop(['ever_married', 'work_type'], axis=1)
df.to_csv("stroke_data_reduced.csv", index=False)