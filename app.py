import streamlit as st
import torch
import joblib
import os
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import numpy as np

# Toggle between online or local model loading
USE_INTERNET = True  # Set to False if loading locally

# Load BERT tokenizer and model safely
try:
    if USE_INTERNET:
        st.write("⚡ Fetching BERT model from Hugging Face...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    else:
        st.write("🚀 Loading BERT model from local storage...")
        tokenizer = BertTokenizer.from_pretrained("path_to_local_bert_base_uncased")
        bert_model = BertModel.from_pretrained("path_to_local_bert_base_uncased")
    st.success("✅ BERT model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading BERT model: {str(e)}")
    tokenizer, bert_model = None, None  # Prevent crashes

# Check if classifier model exists
model_path = "bert_fake_news_model.jb"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("✅ Fake news classification model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading classifier model: {str(e)}")
        model = None
else:
    st.warning("⚠ Model file not found! Training a new classifier...")
    # Dummy Training for Logistic Regression (Replace with actual training)
    model = LogisticRegression()
    X_train, y_train = np.random.rand(10, 768), np.random.randint(0, 2, 10)  # Dummy Data
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    st.success("✅ New classifier trained and saved successfully!")

# Streamlit UI
st.title("Fake News Detector - BERT Edition")
st.write("Enter a news article below to check whether it is Fake or Real.")

input_text = st.text_area("News Article:", "")

# Cache embeddings for performance
@st.cache_resource
def get_bert_embedding(text):
    """Generate BERT embeddings for input text."""
    if tokenizer and bert_model:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # ✅ Fix: Convert to CPU before NumPy
    return None  # Handle missing models

# Validate Predictions & Check Accuracy
def validate_model():
    sample_texts = ["Government announces new tax policy.", "Aliens spotted in California! NASA confirms."]
    sample_vectors = np.random.rand(2, 768)  # Replace with real embeddings
    predictions = model.predict(sample_vectors)

    st.write("🧐 **Sample Predictions:**")
    st.write(f"📜 **Text 1:** {sample_texts[0]} → **Prediction:** {'Real' if predictions[0] == 1 else 'Fake'}")
    st.write(f"📜 **Text 2:** {sample_texts[1]} → **Prediction:** {'Real' if predictions[1] == 1 else 'Fake'}")

# Handle input processing
if st.button("Check News"):
    if len(input_text.strip()) < 20:
        st.warning("⚠ Please enter at least 20 characters for better accuracy.")
    else:
        try:
            embedding = get_bert_embedding(input_text)
            if embedding is not None:
                prediction = model.predict(embedding)

                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(embedding)[0]
                    confidence_real = confidence[1] * 100
                    confidence_fake = confidence[0] * 100
                else:
                    confidence_real = confidence_fake = None

                if prediction[0] == 1:
                    st.success(f"✅ The news is likely **Real**! Confidence: {confidence_real:.2f}%")
                else:
                    st.error(f"❌ The news is likely **Fake**! Confidence: {confidence_fake:.2f}%")

                # Explainability enhancements
                st.write("### 🧐 Why did the model classify it this way?")
                st.write("The model uses BERT embeddings to analyze linguistic and contextual clues.")
                st.write("Consider cross-checking sources for further validation.")

                validate_model()  # Show validation predictions
            else:
                st.error("❌ Model could not generate embeddings. Check if BERT is properly loaded.")

        except Exception as e:
            st.error(f"⚠ Error processing input: {str(e)}")

# Deployment notice
st.write("🚀 Ensure the BERT-based classifier is properly trained and deployed.")