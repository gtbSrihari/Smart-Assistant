import streamlit as st
from summarizer import generate_summary
from qa import answer_question
from challenge import generate_question, evaluate_answer
from utils import extract_pdf_text, extract_txt_text
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_qg():
    tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-qg-hl")
    model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-qg-hl")
    return tokenizer, model

st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_type = uploaded_file.type
    with st.spinner("Extracting text..."):
        if file_type == "application/pdf":
            full_text = extract_pdf_text(uploaded_file)
        else:
            full_text = extract_txt_text(uploaded_file)

    st.subheader("Summary")
    with st.spinner("Summarizing document..."):
        summary = generate_summary(full_text)
    st.write(summary)

    mode = st.radio("Choose Interaction Mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        st.subheader("Ask Anything")
        question = st.text_input("Your Question:")
        if question:
            qa = load_qa()
            with st.spinner("Thinking..."):
                answer, confidence = answer_question(question, full_text)
            st.success(f"Answer: {answer}")
            st.caption(f"Confidence: {confidence:.2f}")

    elif mode == "Challenge Me":
        st.subheader("🧩 Challenge Me")
        qa = load_qa()
        tokenizer, model = load_qg()

        with st.spinner("Generating questions..."):
            answers_to_use = []
            for sent in full_text.split(". "):
                if len(sent.split()) > 8:
                    ans = sent.strip().split()[1]
                    if ans.isalpha():
                        answers_to_use.append(ans)
                if len(answers_to_use) >= 3:
                    break

            questions = [generate_question(full_text[:1000], ans) for ans in answers_to_use]

        user_answers = []
        for i, q in enumerate(questions):
            ans = st.text_input(f"Q{i+1}: {q}")
            user_answers.append(ans)

        if st.button("Evaluate Answers"):
            st.subheader("📊 Results")
            for i, user_ans in enumerate(user_answers):
                evaluation = evaluate_answer(user_ans, answers_to_use[i])
                st.write(f"**Q{i+1}**: {evaluation}")