import pickle
import requests
from flask import Flask, render_template, request

def load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}:", e)
        return None

vectorizer = load("vectorizer.pkl")
lr_model   = load("lr_model.pkl")
rf_model   = load("rf_model.pkl")
gb_model   = load("gb_model.pkl")
dt_model   = load("dt_model.pkl")

API_KEY = ""

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def label(pred):
    return "Not A Fake News" if int(pred) == 1 else "Fake News"

def predict_ml(news):

    if not vectorizer:
        return {
            "LR": "Uncertain",
            "RF": "Uncertain",
            "GB": "Uncertain",
            "DT": "Uncertain"
        }

    X = vectorizer.transform([news])

    return {
        "LR": label(lr_model.predict(X)[0]),
        "RF": label(rf_model.predict(X)[0]),
        "GB": label(gb_model.predict(X)[0]),
        "DT": label(dt_model.predict(X)[0]),
    }

def call_ai(prompt):

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=15
        )

        print("Status:", response.status_code)
        print("Response:", response.text)

        if response.status_code == 200:
            data = response.json()

            return data["choices"][0]["message"]["content"]

    except Exception as e:
        print("AI Error:", e)

    return None

def ai_check(news):

    prompt = f"""
Decide whether this news is REAL or FAKE.

Reply ONLY with:
REAL
or
FAKE

News:
{news}
"""

    result = call_ai(prompt)

    if not result:
        return "Unavailable"

    result = result.strip().lower()

    if "real" in result:
        return "Not A Fake News"

    elif "fake" in result:
        return "Fake News"

    return "Uncertain"

def ai_explain(news):

    prompt = f"""
Explain in 2 short lines whether this news is real or fake.

News:
{news}
"""

    result = call_ai(prompt)

    return result if result else "AI unavailable"

def final_decision(ml, ai):

    votes = list(ml.values())

    if ai in ["Not A Fake News", "Fake News"]:
        votes.append(ai)

    real = votes.count("Not A Fake News")
    fake = votes.count("Fake News")

    total = len(votes)

    if real > fake:
        final = "Not A Fake News"
        conf = int((real / total) * 100)

    elif fake > real:
        final = "Fake News"
        conf = int((fake / total) * 100)

    else:
        final = "Uncertain"
        conf = 50

    return final, conf

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/check", methods=["GET", "POST"])
def check():

    if request.method == "GET":
        return render_template(
            "check.html",
            result=None,
            explanation=None,
            news=""
        )

    if request.form.get("clear"):
        return render_template(
            "check.html",
            result=None,
            explanation=None,
            news=""
        )

    news = request.form.get("news", "").strip()

    if not news:
        return render_template(
            "check.html",
            result=None,
            explanation=None,
            news=""
        )

    ml = predict_ml(news)

    ai = ai_check(news)

    print("AI Result:", ai)

    
    final, conf = final_decision(ml, ai)

    result = {
        **ml,
        "AI": ai,
        "FINAL": final,
        "CONF": conf
    }

    explanation = None

    if request.form.get("explain") and ai != "Unavailable":
        explanation = ai_explain(news)

    return render_template(
        "check.html",
        result=result,
        explanation=explanation,
        news=news
    )

if __name__ == "__main__":
    app.run(debug=True)