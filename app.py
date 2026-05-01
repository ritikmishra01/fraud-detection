from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import get_cleaned_data

app = Flask(__name__)

print("🚀 Starting App...")

# Load data
X_train, X_test, y_train, y_test = get_cleaned_data('creditcard.csv')

# ✅ Better model for imbalance
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")

# 🔧 Threshold for fraud detection (important)
THRESHOLD = 0.3


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

        # Remove unnecessary columns if present
        df = df.drop(columns=[c for c in ['Time', 'Class'] if c in df.columns], errors='ignore')

        # Ensure correct column order
        df = df[X_train.columns]

        # 🔥 Use probability instead of direct prediction
        probs = model.predict_proba(df)[:, 1]
        predictions = (probs > THRESHOLD).astype(int)

        fraud_count = int(predictions.sum())
        total = len(predictions)
        normal_count = total - fraud_count

        # 🔥 Evaluate model on test data with same threshold
        test_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (test_probs > THRESHOLD).astype(int)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        result = f"""
        <b>Total Transactions:</b> {total} <br>
        <b>Fraud Detected:</b> {fraud_count} ❌ <br>
        <b>Normal Transactions:</b> {normal_count} ✅ <br><br>

        <b>Model Performance:</b><br>
        Accuracy: {round(acc*100, 2)}% <br>
        Precision: {round(precision, 3)} <br>
        Recall: {round(recall, 3)} <br>
        F1 Score: {round(f1, 3)}
        """

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {e}")


if __name__ == "__main__":
    print("🔥 Server running at http://127.0.0.1:5000")
    app.run(debug=True)