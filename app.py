import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

def clean(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

st.title("📄 Resume Screening Tool")
resume = st.text_area("Paste your Resume here 👤")
jd = st.text_area("Paste Job Description here 💼")

if st.button("Check Match"):
    if resume and jd:
        res_clean = clean(resume)
        jd_clean = clean(jd)
        vectorizer = TfidfVectorizer()
        vec = vectorizer.fit_transform([res_clean, jd_clean])
        score = cosine_similarity(vec[0:1], vec[1:2])[0][0]
        st.success(f"✅ Match Score: {score:.2f}")
    else:
        st.warning("Please paste both resume and job description!")
