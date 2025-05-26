import streamlit as st
import joblib

# Load models
vectorizer, model = joblib.load("vectorizer.jb"), joblib.load("lr_model.jb")

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Paste a news article below to determine if it's Fake or Real.")

input_text = st.text_area("News Article:", "")

if st.button("🔍 Check News") and input_text.strip():
    prediction = model.predict(vectorizer.transform([input_text]))[0]
    st.success("✅ The News is Real!") if prediction == 1 else st.error("❌ The News is Fake!")
elif st.button("🔍 Check News"):
    st.warning("⚠️ Please enter some text for analysis.")