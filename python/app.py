from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

import pandas as pd
import pickle
import re
import os

app = Flask(__name__)
CORS(app)

# =========================
# OPENAI
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Model loaded")

# =========================
# CLEAN TEXT
# =========================

def clean_text(text):
    text = str(text).lower().strip()

    text = text.replace("ko", "không")
    text = text.replace("k", "không")

    text = re.sub(r"[^\w\s]", "", text)

    return text

# =========================
# LOAD DATASET
# =========================

def load_data():
    return pd.read_csv("dataset_demo.csv")

# =========================
# INIT DATA
# =========================

def init_model():
    global df_data

    df_data = load_data()

    df_data["question"] = (
        df_data["question"]
        .astype(str)
        .apply(clean_text)
    )

    print("Loaded:", len(df_data))

init_model()

# =========================
# DOMAIN CHECK
# =========================

KEYWORDS = [
    "sinh thiết",
    "ung thư",
    "u",
    "bướu",
    "polyp",
    "xét nghiệm",
    "giải phẫu bệnh",
    "cin"
]

def is_medical(text):
    return any(k in text for k in KEYWORDS)

# =========================
# SHORT QUESTION
# =========================

def is_short(text):
    return any(x in text for x in [
        "bao nhiêu",
        "giá",
        "chi phí",
        "bao lâu",
        "nguy hiểm không",
        "có sao không"
    ])

# =========================
# GPT PROMPT
# =========================

SYSTEM_PROMPT = """
Bạn là bác sĩ chuyên ngành giải phẫu bệnh.

Nhiệm vụ:
- Chỉ trả lời các câu hỏi liên quan đến giải phẫu bệnh.
- Trả lời ngắn gọn, dễ hiểu.
- Không bịa thông tin.
- Nếu không chắc thì khuyên người dùng đi khám bác sĩ.
"""

# =========================
# GPT CALL
# =========================

def ask_gpt(msg):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": msg
                }
            ],
            temperature=0.3
        )

        return res.choices[0].message.content

    except Exception as e:
        print("GPT ERROR:", e)

        return "⚠️ AI đang bận, vui lòng thử lại sau."

# =========================
# CHAT API
# =========================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(force=True)

        raw_msg = data.get("message", "")
        raw_last = data.get("last_question", "")

        msg = clean_text(raw_msg)
        last = clean_text(raw_last)

        if msg == "":
            return jsonify({
                "reply": "⚠️ Bạn chưa nhập câu hỏi!",
                "new_context": ""
            })

        # CONTEXT

        if is_short(msg) and last != "":
            final = last + " " + msg
            context = last
        else:
            final = msg
            context = msg

        print("FINAL:", final)

        # DOMAIN CHECK

        if not is_medical(final):
            return jsonify({
                "reply": "Tôi chỉ hỗ trợ về giải phẫu bệnh.",
                "new_context": ""
            })

        # EXACT MATCH

        result = df_data[
            df_data["question"] == final
        ]

        if len(result) > 0:

            answer = result.iloc[0]["answer"]

            return jsonify({
                "reply": answer,
                "new_context": context
            })

        # PREDICT INTENT

        query_vector = vectorizer.transform([final])

        predicted_intent = model.predict(
            query_vector
        )[0]

        print("Intent:", predicted_intent)

        result = df_data[
            df_data["intent"] == predicted_intent
        ]

        if len(result) > 0:

            answer = result.iloc[0]["answer"]

            return jsonify({
                "reply": answer,
                "new_context": context
            })

        # GPT FALLBACK

        reply = ask_gpt(final)

        return jsonify({
            "reply": reply,
            "new_context": context
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "reply": "⚠️ Lỗi server!",
            "new_context": ""
        })

# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
