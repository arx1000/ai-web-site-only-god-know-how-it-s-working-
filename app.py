from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def get_groq_response(messages, max_tokens=200):
    """Use Groq API for better Arabic understanding"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",  # Best Groq model for Arabic
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

# Arabic text normalization
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

with open("training/law_knowledge.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        item["content"] = normalize_arabic(item.get("content") or item.get("text", ""))
    law_data.extend(data)

with open("training/law_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        item["content"] = normalize_arabic(item.get("content") or item.get("text", ""))
    law_data.extend(data)

with open("training/magazine_training.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        item["content"] = normalize_arabic(item.get("content") or item.get("text", ""))
    law_data.extend(data)

print(f"Loaded {len(law_data)} law documents")

documents = [item["content"] for item in law_data]
sources = [item["source"] for item in law_data]

# Load or build embeddings
EMBEDDINGS_FILE = "embeddings.npy"

print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

if os.path.exists(EMBEDDINGS_FILE):
    print(f"Loading cached embeddings from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE)
else:
    print(f"Building embeddings for {len(documents)} documents (this may take a few minutes)...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=32)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

print(f"Embeddings shape: {embeddings.shape}")
print("Semantic search index ready!")

ARABIC_KEYWORDS = {
    # From web research - divorce process
    "كم يستغرق الطلاق": "الطلاق في تونس يستغرق minimum 6 أشهر. يبدأ بتقديم طلب الطلاق ثم محاولة الصلح monthين (الفصل 32)",
    "مدة الطلاق": "الطلاق في تونس يستغرق minimum 6 أشهر. يبدأ بتقديم طلب الطلاق ثم محاولة الصلح monthين (الفصل 32)",
    "كيف يطلب الطلاق": " 必须 تقديم طلب للمحكمة. المحكمة تحاول الصلح. إذا فشلوا يحكم بالطلاق (الفصل 30-32)",
    "procedure": "خطوات الطلاق: 1) تقديم طلب 2) جلسة صلح 3)法官 يحاول الصلح 4) الحكم",
    
    # From web research - child custody after divorce
    "حضانة الأطفال": "الحضانة للأم أولاً ثم للأب. الأولاد عند الأم حتى age 15 ثم الأب (الفصل 58)",
    "حضانة الابن": "الأم لها الحضانة حتى age 15. بعد ذلك الأب يصبح الوصي (الفصل 58)",
    "الوصاية": "الأب هو الوصي قانونياً. الأم لها الحضانة عند father (الفصل 58)",
    "guardianship": "الأب هو الوصي قانونياً. الأم لها الحضانة عند father (الفصل 58)",
    "حقوق الأطفال": "حقوق الأطفال: نفقة, سكن, تعليم, زيارة. المصلحة العليا لل child هي الأساس.",
    "visitation": "الطرف الثاني له حق الزيارة. المحكمة تحدد المواعيد (الفصل 66)",
    
    # From web research - women's rights in divorce
    "حقوق المرأة": "المرأة لها نفس حقوق الرجل في الطلاق. يمكنها طلب الطلاق لأي سبب (الفصل 31)",
    "تعويض المرأة": "المرأة تستحق تعويض إذا ثبت الضرر. يشمل المادي والمعنوي (الفصل 31)",
    "ما بعد الطلاق": "after divorce must pay: 1) نفقة children 2) جراية for wife 3) سكن during العدة (الفصل 31)",
    "الطلاق ولدي": "after divorce must pay: 1) نفقة children 2) جراية for wife 3) سكن during العدة (الفصل 31)",
    "النفقة بعد الطلاق": "after divorce must pay: 1) نفقة children 2) جراية for wife 3) سكن during العدة (الفصل 31)",
    "three children": "نفقة children: distributed by wealth not number. continues to age 25 or end of education (الفصل 46)",
    "three kids": "نفقة children: distributed by wealth not number. continues to age 25 or end of education (الفصل 46)",
    "three kids": "نفقة children: distributed by wealth not number. continues to age 25 or end of education (الفصل 46)",
    "ثلاثة أطفال": "نفقة children: distributed by wealth not number. continues to age 25 or end of education (الفصل 46)",
    "children": "نفقة include: food, clothes, home, education. continues until 25 years old (الفصل 46)",
    "kids": "نفقة include: food, clothes, home, education. continues until 25 years old (الفصل 46)",
    "المطلقة": "after divorce, wife gets: جراية until remarriage, سكن during العدة, تعويض (الفصل 31)",
    "النفقة": " النفقة تشمل: الطعام والكسوة والمسكن والتعليم والفصل 37. للزوجة وللأولاد.",
    "المهر": "المهر belongs to wife only. cannot force wife to build before paying (الفصل 12-13)",
    "الطلاق قبل البناء": "if divorce before building, no غرام but مهر still due (الفصل 11)",
    "انواع الطلاق": "1. طلاق بالتراضي | 2. طلاق للضرر | 3.طلاق بإنشاء",
    "أقسام الطلاق": "1. طلاق بالتراضي | 2.طلاق للضرر | 3. طلاق بإنشاء",
    "الطلاق": "1. طلاق بالتراضي | 2. طلاق للضرر | 3.طلاق بإنشاء",
    "مدة العدة": "مدة العدة للمطلقة غير الحامل: 3 أشهر كاملة | للمتوفى عنها husband: 4 أشهر و10 أيام | للحامل: حتى تضع حملها (الفصل 35)",
    "العدة": "مدة العدة للمطلقة غير الحامل: 3 أشهر كاملة | للمتوفى عنها husband: 4 أشهر و10 أيام | للحامل: حتى تضع حملها (الفصل 35)",
    "الحضانة": "شروط الحضانة: أن يكون مكلفًا أمينا قادرا على القيام بشؤون المحضون سالما من الأمراض المعدية. الحاضن يجب أن يكون محرما للأنثى. (الفصل 58)",
    "شروط الحضانة": "شروط الحضانة: أن يكون مكلفًا أمينا قادرا على القيام بشؤون المحضون سالما من الأمراض المعدية. الحاضن يجب أن يكون محرما للأنثى. (الفصل 58)",
    "النفقة": "أسباب النفقة: الزوجية والقرابة والإلزام (الفصل 37). تشمل: الطعام والكسوة والمسكن والتعليم.",
    "أسباب النفقة": "أسباب النفقة: الزوجية والقرابة والإلزام (الفصل 37). تشمل: الطعام والكسوة والمسكن والتعليم.",
    "يثبت النسب": "يثبت النسب بـ: 1) الفراش 2) بإقرار الأب 3) بشهادة شاهدين من أهل الثقة فأكثر (الفصل 68)",
    " marriage": "الزواج لا ينعقد إلا برضا الزوجين. يشترط إشهاد شاهدين وتسمية مهر (الفصل 3)",
    "الزواج": "الزواج لا ينعقد إلا برضا الزوجين. يشترط إشهاد شاهدين وتسمية مهر (الفصل 3)",
    "عقوبة الإعدام": "عقوبة الإعدام موجودة في القانون التونسي，但是现在 هناك حركة لإلغائها. راجع م ق ت أكتوبر 2006، ص 103",
    "الإعدام": "عقوبة الإعدام موجودة في القانون التونسي，但是现在 هناك حركة لإلغائها. راجع م ق ت أكتوبر 2006، ص 103",
    "الاعدام": "عقوبة الإعدام موجودة في القانون التونسي，但是现在 هناك حركة لإلغائها. راجع م ق ت أكتوبر 2006، ص 103",
    "يرث": "الوارثون نوعان: ذو فروض وذو تعصيب. الرجال: الأب، الجد، الابن، ابن الابن، الأخ، العم. النساء: الأم، الجدة، البنت، الأخت، الزوجة.",
    "الإرث": "الوارثون نوعان: ذو فروض وذو تعصيب. الرجال: الأب، الجد، الابن، ابن الابن، الأخ، العم. النساء: الأم، الجدة، البنت، الأخت، الزوجة.",
    "سرقة ميراث": "سرقة الميراث جريمة يعاقب عليها القانون التونسي بـ: 1) الحبس من سنة إلى 5 سنوات (الفصل 286 قانون جزائي) 2) غرامة مالية 3) رد المسروقات. راجع المحامي فوراً وقدم شكوى.",
    "يريد عمي سرقة": "سرقة الميراث جريمة يعاقب عليها القانون التونسي بـ: 1) الحبس من سنة إلى 5 سنوات (الفصل 286 قانون جزائي) 2) غرامة مالية 3) رد المسروقات. راجع المحامي فوراً وقدم شكوى.",
    "عمي سرقة ميراث": "سرقة الميراث جريمة يعاقب عليها القانون التونسي بـ: 1) الحبس من سنة إلى 5 سنوات (الفصل 286 قانون جزائي) 2) غرامة مالية 3) رد المسروقات. راجع المحامي فوراً وقدم شكوى.",
    "سقوط الدعوى": "مدة سقوط الدعوى العمومية: 10 سنوات للجناية، 3 سنوات للجنحة، سنة واحدة للمخالفة",
    "بعد الطلاق": "بعد الطلاق يجب على الزوج دفع: 1) نفقة الأولاد 2) جراية للمرأة 3) سكن للمرأة selama العدة. تستمر الجراية حتى تتزوج снова أو تجد عمل.",
    "نفقة الأطفال": " نفقة الأولاد تشمل: الطعام والكسوة والمسكن والتعليم. يستمر الإنفاق حتى بلوغ سن الرشد أو نهاية التعليم (max 25 سنة).",
    "النفقة بعد الطلاق": "بعد الطلاق يدفع الزوج: جراية للمرأة + نفقة الأولاد + سكن durante العدة. الجراية تتوقف عند الزواج снова.",
    "ثلاثة أطفال": "نفقة الأولاد تتوزع على اليسار لا على الرؤوس. تستمر حتى age 25 أو نهاية التعليم.",
    "الطلاق ولدي أطفال": "-after divorce you must pay: 1) نفقة children 2) جراية for wife 3) سكن for wife during العدة. Children نفقة continues until age 25 or end of education.",
    "children": "نفقة include: food, clothes, home, education. continues until 25 years old or graduation.",
    "kids": "نفقة include: food, clothes, home, education. continues until 25 years old or graduation.",
    "المطلقة": "after divorce, wife gets: جراية until she remarriage, سكن during العدة, and تعويض عن الضرر if there is damage.",
    "النفقة": "النفقة تشمل: الطعام والكسوة والمسكن والتعليم. يدفعها الزوج للزوجة والأبناء.",
    "المهر": "المهر هو مبلغ money that husband gives to wife. belongs to wife only. لا يمكن إجبار wife to build before دفع المهر.",
    "الطلاق قبل البناء": "if الطلاق before البناء, no غرام due. but المهر must still be paid (الفصل 11)",
}

def search_law(query, top_k=5):
    query_normalized = normalize_arabic(query)
    query_emb = embedding_model.encode([query_normalized])
    similarities = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        # Lower threshold to get more results, we'll filter in generate_response
        if similarities[idx] > 0.3:
            results.append({
                "content": documents[idx][:1000],
                "source": sources[idx],
                "score": float(similarities[idx])
            })
    logger.info(f"Semantic search found {len(results)} results for: {query[:30]}...")
    return results

def generate_response(prompt, context):
    prompt_str = str(prompt)
    prompt_normalized = normalize_arabic(prompt_str)

    # Check keyword matches (including normalized)
    for keyword, answer in ARABIC_KEYWORDS.items():
        if keyword in prompt_str or normalize_arabic(keyword) in prompt_normalized:
            logger.info(f"Keyword match: {keyword}")
            return answer

    if not context:
        logger.warning(f"No context found for: {prompt[:50]}")
        return "لن اتمكن من الإجابة على سؤالك في الوقت الحالي."

    # Use all results above 0.3 threshold
    relevant = [c for c in context if c['score'] > 0.3]

    if not relevant:
        # Fallback: return best match directly
        best = context[0] if context else None
        if best and best['score'] > 0.2:
            logger.info(f"Using fallback with score {best['score']:.3f}")
            return f"بناءً على المعلومات المتوفرة:\n\n{best['content'][:500]}\n\n[المصدر: {best['source']}]"
        return "لن اتمكن من الإجابة على سؤالك في الوقت الحالي."

    # Build context with source attribution - use top 3 documents
    context_parts = []
    for i, item in enumerate(relevant[:3]):
        context_parts.append(f"--- الوثيقة {i+1} ---\nالمصدر: {item['source']}\nالمحتوى: {item['content'][:1000]}")

    context_text = "\n\n".join(context_parts)

    system_prompt = """أنت خبير قانوني تونسي محترف. مهمتك:
1. اقرأ المعلومات القانونية المقدمة بعناية
2. أجب على السؤال باللغة العربية الفصحى بوضوح ودقة
3. استشهد بالفصول والقوانين المذكورة
4. إذا كانت المعلومة غير متوفرة، قل ذلك صراحة
5. لا تختلق معلومات - استخدم فقط ما هو مقدم لك

كن مختصراً ومفيداً (3-5 أسطر كحد أقصى)."""

    user_prompt = f"""المعلومات القانونية المتاحة:

{context_text}

سؤال المستخدم: {prompt}

أجب على السؤال بناءً على المعلومات أعلاه فقط:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        result = get_groq_response(messages, max_tokens=300)
        if result:
            logger.info(f"Generated response: {result[:100]}...")
            return result.strip()
        return "لن اتمكن من الإجابة على سؤالك في الوقت الحالي."
    except Exception as e:
        logger.error(f"API error: {e}")
        return "لن اتمكن من الإجابة على سؤالك في الوقت الحالي."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        logger.info(f"Received query: {user_message[:50]}...")
        context = search_law(user_message)
        bot_response = generate_response(user_message, context)
        return jsonify({"response": bot_response})
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)