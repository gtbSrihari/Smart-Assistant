from pdfminer.high_level import extract_text
import tempfile
import os

def extract_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        text = extract_text(tmp_file.name)
    os.unlink(tmp_file.name)
    return text

def extract_txt_text(uploaded_file):
    return uploaded_file.read().decode("utf-8")
