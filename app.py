from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
import re
import logging
import requests
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

def get_groq_response(messages, max_tokens=300):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"Groq API error: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Groq API exception: {e}")
        return None

def normalize_arabic(text):
    if not text:
        return ""
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Loading law data...")
law_data = []

for f in sorted(glob.glob("training/split/*.json")):
    try:
        print(f"  Loading {f}...")
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
            for item in data:
                item["content"] = normalize_arabic(item.get("content", ""))
            law_data.extend(data)
        print(f"  Loaded {f}: {len(data)} items")
    except Exception as e:
        print(f"  Error loading {f}: {e}")

print(f"Loaded {len(law_data)} documents")

documents = [item["content"] for item in law_data]
sources = [item["source"] for item in law_data]

def simple_search(query, top_k=3):
    query_norm = normalize_arabic(query).lower()
    query_words = set(query_norm.split())
    
    scores = []
    for i, doc in enumerate(documents):
        score = 0
        doc_norm = doc.lower()
        for word in query_words:
            if word in doc_norm:
                score += 1
        if score > 0:
            scores.append((i, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scores[:top_k]:
        results.append({
            "content": documents[idx][:1500],
            "source": sources[idx],
            "score": score
        })
    return results

ARABIC_KEYWORDS = {
    "كم يستغرق الطلاق": "الطلاق في تونس يستغرق minimum 6 أشهر. يبدأ بتقديم طلب الطلاق ثم محاولة الصلح شهرين (الفصل 32)",
    "مدة الطلاق": "الطلاق في تونس يستغرق minimum 6 أشهر. يبدأ بتقديم طلب الطلاق ثم محاولة الصلح شهرين (الفصل 32)",
    "كيف يطلب الطلاق": "تقديم طلب للمحكمة. المحكمة تحاول الصلح. إذا فشلوا يحكم بالطلاق (الفصل 30-32)",
    "حضانة الأطفال": "الحضانة للأم أولاً ثم للأب. الأولاد عند الأم حتى age 15 ثم الأب (الفصل 58)",
    "حضانة الابن": "الأم لها الحضانة حتى age 15. بعد ذلك الأب يصبح الوصي (��لفصل 58)",
    "حقوق الأطفال": "حقوق الأطفال: نفقة, سكن, تعليم, زيارة. المصلحة العليا لل child هي الأساس.",
    "visitation": "الطرف الثاني له حق الزيارة. المحكمة تحدد المواعيد (الفصل 66)",
    "حقوق المرأة": "المرأة لها نفس حقوق الرجل في الطلاق. يمكنها طلب الطلاق لأي سبب (الفصل 31)",
    "تعويض المرأة": "المرأة تستحق تعويض إذا ثبت الضرر. يشمل المادي والمعنوي (الفصل 31)",
    "النفقة": "النفقة تشمل: الطعام والكسوة والمسكن والتعليم. يدفعها الزوج للزوجة والأبناء.",
    "المهر": "المهر belongs to wife only. cannot force wife to build before paying (الفصل 12-13)",
    "الطلاق": "1. طلاق بالتراضي | 2. طلاق للضرر | 3.طلاق بإنشاء",
    "العدة": "مدة العدة للمطلقة غير الحامل: 3 أشهر كاملة | للمتوفى عنها husband: 4 أشهر و10 أيام (الفصل 35)",
    "الزواج": "الزواج لا ينعقد إلا برضا الزوجين. يشترط إشهاد شاهدين وتسمية مهر (الفصل 3)",
    "الإرث": "الوارثون نوعان: ذو فروض وذو تعصيب. الرجال: الأب، الجد، الابن، ابن الابن، الأخ، العم.",
    "المطلقة": "after divorce, wife gets: جراية until she remarriage, سكن during العدة, تعويض (الفصل 31)",
    "نفقة الأطفال": "نفقة الأولاد تشمل: الطعام والكسوة والمسكن والتعليم. يستمر حتى بلوغ سن الرشد.",
    "الطلاق بعد": "بعد الطلاق يجب على الزوج دفع: 1) نفقة الأولاد 2) جراية للمرأة 3) سكن durante العدة",
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        logger.info(f"Query: {user_message[:50]}...")
        
        # Check keyword matches first
        user_norm = normalize_arabic(user_message).lower()
        for keyword, answer in ARABIC_KEYWORDS.items():
            if keyword in user_norm or normalize_arabic(keyword).lower() in user_norm:
                logger.info(f"Keyword match: {keyword}")
                return jsonify({"response": answer})
        
        # Simple text search
        context = simple_search(user_message, top_k=3)
        
        if not context:
            return jsonify({"response": "لن اتمكن من الإجابة على سؤالك. يرجى إعادة الصياغة."})
        
        context_parts = []
        for i, item in enumerate(context):
            context_parts.append(f"--- الوثيقة {i+1} ---\nالمصدر: {item['source']}\nالمحتوى: {item['content'][:800]}")
        
        context_text = "\n\n".join(context_parts)
        
        system_prompt = """أنت خبير قانوني تونسي محترف. أجب بالعربية الفصحى بوضوح.
استشهد بالفصول والقوانين المذكورة. كن مختصراً ومفيداً."""
        
        user_prompt = f"""المعلومات القانونية:

{context_text}

سؤال المستخدم: {user_message}

أجب بناءً على المعلومات أعلاه:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        result = get_groq_response(messages, max_tokens=300)
        
        if result:
            return jsonify({"response": result.strip()})
        return jsonify({"response": "لن اتمكن من الإجابة على سؤالك في الوقت الحالي."})
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)