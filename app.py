import streamlit as st
import joblib
import os

# Cache model loading for efficiency
@st.cache_resource
def load_models():
    """Load vectorizer and classification model."""
    if os.path.exists("vectorizer.jb") and os.path.exists("lr_model.jb"):
        return joblib.load("vectorizer.jb"), joblib.load("lr_model.jb")
    else:
        st.error("❌ Model files not found! Ensure 'vectorizer.jb' and 'lr_model.jb' exist.")
        return None, None

vectorizer, model = load_models()

# Streamlit UI Setup
st.title("📰 Fake News Detector")
st.write("Paste a news article below to determine if it's **Fake or Real** using AI.")

# User Input Section
user_input = st.text_area("Enter the news text:", height=150)

# Prediction Logic
if st.button("🔍 Check News"):
    if user_input.strip():
        if vectorizer and model:
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]

            if prediction == 1:
                st.success("✅ The News is **Real**!")
            else:
                st.error("❌ The News is **Fake**!")
        else:
            st.error("⚠ Model not loaded. Ensure the required files exist.")
    else:
        st.warning("⚠ Please enter some text for analysis.")

st.info("🔎 This AI tool helps detect fake news using text analysis and machine learning.")
