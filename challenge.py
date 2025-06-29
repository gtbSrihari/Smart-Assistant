from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load T5 QG model
qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")

def generate_question(context, answer):
    input_text = f"highlight: {context.replace(answer, f'<hl> {answer} <hl>')}"
    inputs = qg_tokenizer(input_text, return_tensors="pt")
    outputs = qg_model.generate(inputs["input_ids"])
    return qg_tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_answer(user_ans, correct_ans):
    vectorizer = TfidfVectorizer().fit_transform([user_ans, correct_ans])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    if similarity > 0.6:
        return f"✅ Correct! (Similarity: {similarity:.2f})"
    else:
        return f"❌ Incorrect. Correct: {correct_ans} (Similarity: {similarity:.2f})"
