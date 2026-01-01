from flask import Flask, render_template, request
import pickle
import numpy as np
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

from database import init_db, get_db_connection


app = Flask(__name__)

init_db()


# Load models
lstm_model = load_model(r"C:\Users\Pc\OneDrive\Desktop\Alex..New..Folder\Deep_Learning\NLP_Project\Notebooks\models\lstm_model.h5")

tokenizer = pickle.load(open(r"C:\Users\Pc\OneDrive\Desktop\Alex..New..Folder\Deep_Learning\NLP_Project\Notebooks\models\tokenizer.pkl", "rb"))
label_encoder = pickle.load(open(r"C:\Users\Pc\OneDrive\Desktop\Alex..New..Folder\Deep_Learning\NLP_Project\Notebooks\models\label_encoder.pkl", "rb"))

MAX_LEN = 150
CATEGORIES = list(label_encoder.classes_)

zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    framework="pt"
)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def normalize_category(cat):
    cat = cat.lower()

    if "money transfer" in cat:
        return "Money transfer"
    elif "checking" in cat or "savings" in cat or "bank account" in cat:
        return "Checking or savings account"
    elif "debt" in cat:
        return "Debt collection"
    elif "mortgage" in cat:
        return "Mortgage"
    elif "credit" in cat:
        return "Credit reporting"
    else:
        return "Other"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        text = request.form["complaint"]

        tf_result = zero_shot(text, CATEGORIES)
        raw_tf_cat = tf_result['labels'][0]
        tf_score = tf_result['scores'][0]   # ⭐ confidence
        tf_cat = normalize_category(raw_tf_cat)

        seq = tokenizer.texts_to_sequences([clean_text(text)])
        pad = pad_sequences(seq, maxlen=MAX_LEN)

        lstm_probs = lstm_model.predict(pad)[0]
        lstm_index = np.argmax(lstm_probs)
        lstm_cat = label_encoder.inverse_transform([lstm_index])[0]
        lstm_score = float(lstm_probs[lstm_index])  # ⭐ confidence


        sentiment = sentiment_pipe(text)[0]['label']
        priority = "High" if sentiment == "NEGATIVE" else "Medium"
        
        # Ensemble confidence
        confidence = (tf_score + lstm_score) / 2

        # Bonus if both models agree
        if tf_cat == lstm_cat:
            confidence += 0.1

        confidence = round(min(confidence, 0.99), 2)

        root_map = {
            "Checking or savings account": "Unauthorized transaction",
            "Debt collection": "Incorrect debt collection",
            "Money transfer": "Transaction failure",
            "Mortgage": "Loan servicing issue",
            "Credit reporting": "Incorrect credit report",
            "Other": "General service issue"
        }

        reply_map = {
            "Checking or savings account":
            "We apologize for the unauthorized transaction. Our team is reviewing your account.",

            "Debt collection":
            "We apologize for the debt collection issue. Our team is investigating.",

            "Money transfer":
            "We regret the inconvenience caused by the transfer issue. Our team is actively working on it.",

            "Mortgage":
            "We are reviewing your mortgage-related concern and will update you shortly.",

            "Credit reporting":
            "We apologize for the incorrect credit reporting. Our team is addressing this.",

            "Other":
            "Thank you for bringing this issue to our attention. Our support team is reviewing it."
        }

        result = {
            "Complaint": text,
            "Transformer Category": tf_cat,
            "LSTM Category": lstm_cat,
            "Sentiment": sentiment,
            "Priority": priority,
            "Root Cause": root_map.get(tf_cat),
            "Automated Reply": reply_map.get(tf_cat),
            "Confidence": confidence
        }
        
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO complaints (
            complaint_text,
            transformer_category,
            lstm_category,
            sentiment,
            priority,
            confidence,
            root_cause,
            automated_reply
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            text,
            tf_cat,
            lstm_cat,
            sentiment,
            priority,
            result.get("Confidence"),
            result.get("Root Cause"),
            result.get("Automated Reply")
        ))

        conn.commit()
        conn.close()
        
        print("Saved successfully!")



    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
