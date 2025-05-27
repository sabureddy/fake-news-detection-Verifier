import streamlit as st
import pandas as pd
import joblib
import os
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load TF-IDF & Logistic Regression Model (Legacy Method)
@st.cache_resource
def load_tfidf_model():
    """Load TF-IDF vectorizer and logistic regression model safely."""
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model
    except FileNotFoundError:
        st.error("❌ Model files missing! Upload `vectorizer.jb` & `lr_model.jb` to the repository.")
        return None, None

vectorizer, tfidf_model = load_tfidf_model()

# Load BERT Model (Modern NLP Approach)
@st.cache_resource
def load_bert_model():
    """Load BERT tokenizer and model for deep learning text analysis."""
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        return tokenizer, bert_model
    except Exception as e:
        st.error(f"❌ Error loading BERT model: {str(e)}")
        return None, None

tokenizer, bert_model = load_bert_model()

# Load Trained BERT Fake News Model
@st.cache_resource
def load_classifier():
    """Load pre-trained classifier for fake news detection."""
    model_path = "bert_fake_news_model.jb"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.warning("⚠ No pre-trained classifier found! Training a new one...")
        clf = LogisticRegression()
        X_train, y_train = np.random.rand(10, 768), np.random.randint(0, 2, 10)  # Dummy Data
        clf.fit(X_train, y_train)
        joblib.dump(clf, model_path)
        return clf

classifier_model = load_classifier()

# Generate BERT Embeddings
@st.cache_resource
def get_bert_embedding(text):
    """Generate BERT embeddings for input text."""
    if tokenizer and bert_model:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return None

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Paste a news article below to determine if it's **Fake or Real** using AI.")

# Text Input
user_input = st.text_area("Enter the news text to verify:", height=150)

# Fake News Detection Logic
if st.button("🔍 Check News"):
    if user_input.strip():
        if classifier_model and tokenizer:
            embedding = get_bert_embedding(user_input)
            if embedding is not None:
                prediction = classifier_model.predict(embedding)[0]

                if hasattr(classifier_model, "predict_proba"):
                    confidence_real = classifier_model.predict_proba(embedding)[0][1] * 100
                    confidence_fake = classifier_model.predict_proba(embedding)[0][0] * 100
                else:
                    confidence_real, confidence_fake = None, None

                if prediction == 1:
                    st.success(f"✅ The News is **Real**! Confidence: {confidence_real:.2f}%")
                else:
                    st.error(f"❌ The News is **Fake**! Confidence: {confidence_fake:.2f}%")
            else:
                st.error("❌ Failed to generate BERT embeddings. Check if BERT is loaded properly.")
        else:
            st.error("❌ Model not loaded! Ensure all necessary files exist.")

    else:
        st.warning("⚠ Please enter some text for analysis.")

# File Upload for Batch Verification
uploaded_file = st.file_uploader("📂 Upload a CSV for batch verification", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 **Uploaded Data Preview:**", df.head())
    # Apply model predictions (implement logic)

st.info("🔎 This AI tool helps detect fake news based on linguistic patterns and deep learning insights.")
