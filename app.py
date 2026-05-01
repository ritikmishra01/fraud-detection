@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return render_template('index.html', prediction_text="❌ Please upload a CSV file")

        df = pd.read_csv(file)

        # Remove unwanted columns
        df = df.drop(columns=[c for c in ['Time', 'Class'] if c in df.columns], errors='ignore')

        # Match training columns
        df = df[X_train.columns]

        # 🔥 Predict probabilities
        probs = model.predict_proba(df)

        # ✅ FIX: handle single-class safely
        if probs.shape[1] < 2:
            return render_template(
                'index.html',
                prediction_text="❌ Model trained on single class. Fix dataset."
            )

        probs = probs[:, 1]

        predictions = (probs > THRESHOLD).astype(int)

        fraud_count = int(predictions.sum())
        total = len(predictions)
        normal_count = total - fraud_count

        # Evaluate model
        test_probs = model.predict_proba(X_test)

        if test_probs.shape[1] < 2:
            return render_template(
                'index.html',
                prediction_text="❌ Test data also has single class issue."
            )

        y_pred = (test_probs[:, 1] > THRESHOLD).astype(int)

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
