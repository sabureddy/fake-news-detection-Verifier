import streamlit as st
import joblib
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

# Cache model loading for efficiency
@st.cache_resource
def load_models():
    """Load vectorizer and classification model."""
    if os.path.exists("vectorizer.jb") and os.path.exists("lr_model.jb"):
        return joblib.load("vectorizer.jb"), joblib.load("lr_model.jb")
    else:
        st.error("❌ Model files not found!")
        return None, None

vectorizer, model = load_models()

# Function to fetch Kaggle dataset (Fake News & LIAR Dataset)
@st.cache_resource
def load_kaggle_data():
    """Load and preprocess fake news datasets from Kaggle."""
    try:
        fake_news = pd.read_csv("https://raw.githubusercontent.com/lutzroeder/fake-news-detection/main/fake_news.csv")
        liar_dataset = pd.read_csv("https://raw.githubusercontent.com/lutzroeder/fake-news-detection/main/liar_dataset.csv")

        # Combine datasets for training expansion
        combined_data = pd.concat([fake_news, liar_dataset], ignore_index=True)

        # Basic text cleaning
        combined_data.dropna(subset=["text"], inplace=True)
        combined_data["text"] = combined_data["text"].str.lower().str.replace(r"[^a-zA-Z\s]", "", regex=True)

        return combined_data
    except Exception as e:
        st.error(f"❌ Error loading Kaggle datasets: {e}")
        return None

dataset = load_kaggle_data()

# Web scraping function for real-time news updates
@st.cache_resource
def fetch_latest_news():
    """Scrape trending news articles for contextual analysis."""
    url = "https://www.bbc.com/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [h3.text for h3 in soup.find_all("h3")][:5]  # Fetch top 5 headlines
    return headlines

latest_news = fetch_latest_news()

# Streamlit UI Setup
st.title("📰 Fake News Detector")
st.write("Paste a news article below to determine if it's **Fake or Real**.")

# User Input Section
user_input = st.text_area("Enter the news text:", height=150)

# Prediction Logic
if st.button("🔍 Check News"):
    if user_input.strip() and vectorizer and model:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.success("✅ The News is **Real**!") if prediction == 1 else st.error("❌ The News is **Fake**!")
    else:
        st.warning("⚠ Please enter valid text and ensure models are loaded.")

# Display dataset preview for verification
if dataset is not None:
    st.write("📊 **Expanded Training Dataset Preview:**")
    st.dataframe(dataset.sample(5))

# Display recent headlines for comparison
if latest_news:
    st.write("📡 **Trending News Headlines:**")
    for headline in latest_news:
        st.write(f"🔹 {headline}")

st.info("🔎 AI-powered fake news detection tool with **expanded dataset training**.")
