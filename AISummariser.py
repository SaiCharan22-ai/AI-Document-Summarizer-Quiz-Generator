import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

st.set_page_config(
    page_title="AI Document Summarizer",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
background: linear-gradient(-45deg,#667eea,#764ba2,#6dd5fa,#2980b9);
background-size:400% 400%;
animation:gradient 15s ease infinite;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.block-container{
background:rgba(255,255,255,0.90);
padding:2rem;
border-radius:20px;
box-shadow:0px 8px 20px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;color:#1E3A8A;font-size:48px;'>

🤖 AI Document Summarizer

</h1>

<h3 style='text-align:center;color:#374151;'>

Quiz Generator for Students

</h3>

""", unsafe_allow_html=True)


# The required models for the application are loaded here
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    quiz_generator = pipeline("text2text-generation", model="google/flan-t5-large")
    return summarizer, quiz_generator

summarizer, quiz_generator = load_models()

#Here text is extracted from PDF and also handles errors
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
    except Exception:
        return ""

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# In this summary is generated from the extracted text
def generate_summary(text):
    text = text[:4000]  # limit for faster processing
    result = summarizer(text, max_length=180, min_length=80, do_sample=False)
    return result[0]["summary_text"]

# Now in this we generate key points
def generate_key_points(summary):
    prompt = f"""
Extract FIVE key points from the summary.

Format EXACTLY like this:

1. <point 1>
2. <point 2>
3. <point 3>
4. <point 4>
5. <point 5>

Summary:
{summary}
"""
    result = quiz_generator(
        prompt,
        max_length=350,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    key_points = result[0]["generated_text"]

    # In this the post processing is done for clean markdown bullets
    lines = [line.strip() for line in key_points.split("\n") if line.strip()]
    formatted_points = "\n".join([f"- {line}" for line in lines])
    return formatted_points

# This generates multiple choice quiz questions 
def generate_quiz(summary):
    prompt = f"""
Generate THREE multiple-choice questions based ONLY on the summary.

Format your output EXACTLY like this:

Q1: <question>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <correct option>

Q2: <question>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <correct option>

Q3: <question>
A) <option>
B) <option>
C) <option>
D) <option>
Answer: <correct option>

Summary:
{summary}
"""
    result = quiz_generator(
        prompt,
        max_length=700,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    quiz_text = result[0]["generated_text"]

    # Here post processing is done for clean spacing
    questions = [q.strip() for q in quiz_text.split("\n") if q.strip()]
    formatted_quiz = "\n\n".join(questions)
    return formatted_quiz

# At this point the PDF is uploaded
st.markdown("## 📄 Upload Academic PDF")

uploaded_file = st.file_uploader(
    "Choose your PDF",
    type="pdf"
)

if uploaded_file is not None:
    st.success("Document uploaded successfully")
    text = extract_text_from_pdf(uploaded_file)

    if len(text) > 0:
        reader = PdfReader(uploaded_file)
        st.subheader("📊 Document Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
         st.metric("Pages", len(reader.pages))

        with col2:
         st.metric("Words", len(text.split()))

        with col3:
         st.metric("Characters", len(text))
    if st.button("🚀 Generate Summary & Quiz"):
            # Now here we generate summary
            with st.spinner("🤖 AI is generating the summary..."):
                summary = generate_summary(text)
            with st.expander("📄 Summary", expanded=True):
                st.write(summary)

            # The key points are generated
            with st.spinner("⭐ Extracting key points..."):
                points = generate_key_points(summary)
            with st.expander("⭐ Key Points"):
                st.markdown(points)

            # Here the quiz get generated
            with st.spinner("📝 Creating quiz questions..."):
                quiz = generate_quiz(summary)
            with st.expander("❓ Quiz Questions"):
                st.markdown(f"```text\n{quiz}\n```")

            # The download button
            output = f"""
SUMMARY

{summary}

KEY POINTS

{points}

QUIZ

{quiz}
"""
            st.download_button(
                "Download Results",
                output,
                file_name="summary_and_quiz.txt"
            )
    else:
        st.error("No text could be extracted from the PDF.")
