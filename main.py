import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2 as pdf
import os
HF_TOKEN = os.getenv("HF_TOKEN")

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(token=HF_TOKEN)

# ----------------------------------------------------
# FRONTEND STYLING
# ----------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: black;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(138, 43, 226, 0.5) 0%, transparent 40%),
        radial-gradient(circle at 75% 60%, rgba(0, 191, 255, 0.5) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(255, 20, 147, 0.3) 0%, transparent 60%),
        url("https://www.transparenttextures.com/patterns/stardust.png");
    background-size: cover;
    background-attachment: fixed;
    color: white !important;
}

h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stTextInput label {
    color: white !important;
}

/* ✔ FIXED — Make textarea visible */
.stTextArea textarea {
    background: white !important;
    color: black !important;
    border-radius: 10px;
}

.stTextArea textarea::placeholder {
    color: #444 !important;
}

/* ✔ FIXED — Make file uploader visible */
.stFileUploader {
    background: white !important;
    border: 2px solid #ddd !important;
    padding: 12px !important;
    border-radius: 10px !important;
}

.stFileUploader label {
    color: black !important;
}

.stFileUploader input {
    color: black !important;
}

.stFileUploader button {
    background: linear-gradient(45deg, #6a0dad, #000080) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(45deg, #6a0dad, #000080);
    color: white !important;
    border-radius: 30px;
    border: none;
    padding: 10px 25px;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.7);
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 25px;
}

.navitem {
    font-size: 18px;
    font-weight: 600;
    color: #ffffffaa;
    cursor: pointer;
}

.navitem:hover {
    color: white;
    text-shadow: 0 0 10px white;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 50px;
    opacity: 0.6;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# NAVBAR
# ----------------------------------------------------
st.markdown("""
<div class="navbar">
    <span class="navitem">Home</span>
    <span class="navitem">About</span>
    <span class="navitem">Contact</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# PDF TEXT EXTRACTION
# ----------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ----------------------------------------------------
# CALL HUGGINGFACE MODEL
# ----------------------------------------------------
def analyze_resume(jd_text, resume_text):
    prompt = f"""
You are an expert ATS scanner and HR recruiter.

Compare the resume with the job description and provide a detailed ATS evaluation.

### Return output ONLY in this format:

## ⭐ ATS Score (Out of 100)

## ✅ What the Resume Did Right

## ❌ What the Resume Did Wrong

## 🔑 Missing Important Keywords

## 🛠 Actionable Improvements

---

Job Description:
{jd_text}

Resume:
{resume_text}
"""

    messages = [
        {"role": "system", "content": "You are an ATS and HR evaluation expert."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1500,
            temperature=0.4,
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"


# ----------------------------------------------------
# MAIN UI
# ----------------------------------------------------
st.title("📄 AI ATS Resume Checker")
st.write("Upload your resume and paste a Job Description to get your ATS score + Wrong & Right analysis.")

jd = st.text_area("Paste Job Description Here", height=200)
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if st.button("Analyze Resume"):
    if not jd:
        st.error("Please paste a Job Description.")
    elif not uploaded_file:
        st.error("Please upload a Resume PDF.")
    else:
        with st.spinner("Analyzing resume using AI..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            result = analyze_resume(jd, resume_text)

        st.subheader("📊 ATS Evaluation Report")
        st.markdown(result)

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("""
<div class="footer">
    © 2026 • AI ATS Resume Checker • All Rights Reserved
</div>
""", unsafe_allow_html=True)