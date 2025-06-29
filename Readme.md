# Smart Assistant for Research Summarization

This project is a lightweight AI-powered assistant that reads documents (PDF/TXT), auto-generates summaries, answers questions based on the document, and challenges the user with logic-based questions. It uses NLP models under the hood to provide an interactive, intelligent reading experience.

---

## Features

- Upload PDF or TXT documents
- Automatic summarization (≤150 words)
- Ask questions based on the document content
- Challenge Mode: auto-generated logic/comprehension questions
- Answer evaluation with similarity feedback

---

## Project Structure

```
smart_assistant/
├── app.py                # Main Streamlit app logic
├── summarizer.py         # Document summarization
├── qa.py                 # Q&A from document context
├── challenge.py          # Question generation and answer evaluation
├── utils.py              # File reading utilities (PDF/TXT)
├── requirements.txt      # Python dependencies
└── README.md             # You are here!
```

---

## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/gtbSrihari/Smart-Assistant.git
cd smart-assistant-research
```

2. **Create a Virtual Environment**

```bash
python -m venv assistant_env
source assistant_env/bin/activate  # or assistant_env\Scripts\activate on Windows
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Run the Application**

```bash
streamlit run app.py
```

---

## Architecture & Reasoning Flow

### Input Phase
- User uploads a `.pdf` or `.txt` file.
- File is processed using `pdfminer.six` or direct read.

### Summarization
- A BART-based transformer model (`distilbart-cnn-12-6`) generates a concise summary.
- Only the first 4000 characters of the doc are summarized to optimize time.

### Question Answering (Ask Anything)
- A DistilBERT QA model (`distilbert-base-uncased-distilled-squad`) processes the user's question and extracts an answer from the **entire document**, not just the summary.

### Challenge Me (Logic Mode)
- A T5-based model (`valhalla/t5-small-qg-hl`) generates 3 logic/comprehension questions from the document using keyword-based highlighting.
- The user answers each question.
- Answers are evaluated using cosine similarity (TF-IDF) to check semantic correctness.

---

## Notes
- Summarization is capped to 150 words for brevity.
- Question evaluation isn't based on keywords alone but contextual similarity.

---

## Future Enhancements
- Long document chunking + multi-turn QA
- Highlighting supporting evidence in document
- Session memory to follow multi-step logic
- Deploy to Hugging Face Spaces

---

## Author

Built by Sri Hari Kande – https://github.com/gtbSrihari?tab=repositories

---

## License

MIT License – feel free to use, modify, and contribute.
