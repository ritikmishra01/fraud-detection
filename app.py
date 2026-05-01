from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import get_cleaned_data
import os

# ✅ FIRST define app
app = Flask(__name__)

print("🚀 Starting App...")

# Load data
X_train, X_test, y_train, y_test = get_cleaned_data()

print("Training class distribution:")
print(y_train.value_counts())

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")

THRESHOLD = 0.3


# ✅ ROUTES AFTER app defined
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return render_template('index.html', prediction_text="❌ Please upload a CSV file")

        df = pd.read_csv(file)

        df = df.drop(columns=[c for c in ['Time', 'Class'] if c in df.columns], errors='ignore')
        df = df[X_train.columns]

        probs = model.predict_proba(df)

        if probs.shape[1] < 2:
            return render_template('index.html',
                prediction_text="❌ Model trained on single class.")

        probs = probs[:, 1]
        predictions = (probs > THRESHOLD).astype(int)

        fraud_count = int(predictions.sum())
        total = len(predictions)
        normal_count = total - fraud_count

        return render_template('index.html',
            prediction_text=f"Fraud: {fraud_count} / {total}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {e}")


# Render config
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
