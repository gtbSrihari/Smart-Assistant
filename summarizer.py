from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    return summarizer(text[:4000], max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
