from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import re
from flask_cors import CORS
import pymysql

# GPT
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")  # 🔥 thay key thật

app = Flask(__name__)
CORS(app)

print("🚀 Loading model...")
model = SentenceTransformer('keepitreal/vietnamese-sbert')

# =========================
# CLEAN TEXT (QUAN TRỌNG)
# =========================
def clean_text(text):
    text = text.lower().strip()

    # normalize cơ bản
    text = text.replace("ko","không").replace("k","không")

    text = re.sub(r'[^\w\s]', '', text)
    return text

# =========================
# LOAD DATA
# =========================
def load_data():
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="chatbot_ai",
        charset="utf8mb4"
    )
    df = pd.read_sql("SELECT message, reply FROM chats", conn)
    conn.close()
    return df

# =========================
# INIT MODEL
# =========================
def init_model():
    global raw_questions, questions, answers, embeddings

    df = load_data()

    raw_questions = df['message'].astype(str).tolist()   # 🔥 giữ bản gốc
    questions = [clean_text(q) for q in raw_questions]
    answers = df['reply'].astype(str).tolist()

    embeddings = model.encode(questions, convert_to_tensor=True)

    print("✅ Loaded:", len(questions))

init_model()

# =========================
# DOMAIN CHECK
# =========================
KEYWORDS = [
    "sinh thiết","ung thư","u","bướu",
    "polyp","xét nghiệm","giải phẫu bệnh"
]

def is_medical(text):
    return any(k in text for k in KEYWORDS)

# =========================
# SHORT QUESTION
# =========================
def is_short(text):
    return any(x in text for x in [
        "bao nhiêu","giá","chi phí",
        "bao lâu","nguy hiểm không","có sao không"
    ])

# =========================
# GPT PROMPT
# =========================
SYSTEM_PROMPT = """
Bạn là bác sĩ chuyên ngành giải phẫu bệnh.

Nhiệm vụ:
- Chỉ trả lời các câu hỏi liên quan đến: bệnh học, ung thư, sinh thiết, xét nghiệm, CIN (loạn sản cổ tử cung), polyp, u bướu.
- Trả lời ngắn gọn, dễ hiểu cho người không chuyên.
- Không sử dụng thuật ngữ quá phức tạp nếu không cần thiết.

Nguyên tắc:
- Không bịa thông tin.
- Nếu câu hỏi không rõ → yêu cầu người dùng nói rõ hơn.
- Nếu không chắc → nói "Tôi chưa đủ thông tin để kết luận, bạn nên đi khám bác sĩ."
- Không trả lời các câu hỏi ngoài lĩnh vực y tế.

Cách trả lời:
- Ưu tiên 2-4 câu.
- Có thể thêm lời khuyên nhẹ (ví dụ: nên đi khám, xét nghiệm).
"""

# =========================
# GPT CALL
# =========================
def ask_gpt(msg):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": msg}
            ],
            temperature=0.3
        )
        return res.choices[0].message.content
    except Exception as e:
        print("GPT ERROR:", e)
        return "⚠️ AI đang bận, thử lại sau."

# =========================
# CHAT API
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)

        raw_msg = data.get("message","")
        raw_last = data.get("last_question","")

        msg = clean_text(raw_msg)
        last = clean_text(raw_last)

        if msg == "":
            return jsonify({
                "reply":"⚠️ Bạn chưa nhập câu hỏi!",
                "new_context":""
            })

        # =========================
        # CONTEXT
        # =========================
        if is_short(msg) and last != "":
            final = last + " " + msg
            context = last
        else:
            final = msg
            context = msg

        print("👉 FINAL:", final)

        # =========================
        # DOMAIN CHECK
        # =========================
        if not is_medical(final):
            return jsonify({
                "reply":"Tôi chỉ hỗ trợ về giải phẫu bệnh.",
                "new_context":""
            })

        # =========================
        # 🔥 1. MATCH CHÍNH XÁC (100%)
        # =========================
        for i, q in enumerate(questions):
            if msg == q:
                print("✅ EXACT MATCH")
                return jsonify({
                    "reply": answers[i],
                    "new_context": context
                })

        # =========================
        # 🔥 2. SBERT
        # =========================
        emb = model.encode(final, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(emb, embeddings)
        scores = scores.cpu().numpy()[0]

        idx = int(np.argmax(scores))
        score = scores[idx]

        print("Score:", score)

        # =========================
        # 🔥 3. HYBRID LOGIC (QUAN TRỌNG)
        # =========================
        if score > 0.75:
            print("👉 SBERT (chắc chắn)")
            reply = answers[idx]

        elif score > 0.55:
            print("👉 SBERT (trung bình)")
            reply = answers[idx]

        else:
            print("👉 GPT (fallback)")
            reply = ask_gpt(final)

        return jsonify({
            "reply": reply,
            "new_context": context
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "reply":"⚠️ Lỗi server!",
            "new_context":""
        })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(port=5000)