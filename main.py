import streamlit as st
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from utils import preprocess_text
from pymongo import MongoClient

# ================= MongoDB Connection ================= #
MONGO_URI = "mongodb://localhost:27017/"  # Update if using MongoDB Atlas
DB_NAME = "sentiment_analysis"
COLLECTION_NAME = "reviews"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# ================= Load Pre-trained Model ================= #
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# ================= Function to Predict Sentiment ================= #
def predict_sentiment(text):
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = torch.argmax(probs, dim=1).item() + 1  # 1 to 5 scale
    return sentiment_score, probs.numpy().tolist()[0]

# ================= Function to Save Analysis in MongoDB ================= #
def save_to_db(review, sentiment, probabilities):
    document = {
        "review": review,
        "sentiment": sentiment,
        "probabilities": probabilities
    }
    collection.insert_one(document)

# ================= Function to Display Past Analyses ================= #
def fetch_past_reviews():
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
    return pd.DataFrame(data)

# ================= Streamlit UI ================= #
st.set_page_config(page_title="Product Review Sentiment Analysis", layout="wide")

st.markdown("<h1 style='text-align: center;'>🛍️ Product Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a product review or upload a CSV/TXT file.</p>", unsafe_allow_html=True)

# ================= User Input Section ================= #
review = st.text_area("Enter your review:", height=70, max_chars=300, placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if review.strip():
        sentiment, probabilities = predict_sentiment(review)

        # Save result in MongoDB
        save_to_db(review, sentiment, probabilities)

        # Display result
        st.markdown(f"<h3 style='text-align: center; color: #007BFF;'>Sentiment Score: {sentiment}</h3>", unsafe_allow_html=True)

        # Plot pie chart
        labels = ["😠 Negative", "😐 Neutral", "😃 Positive"]
        sizes = [sum(probabilities[:2]), probabilities[2], sum(probabilities[3:])]
        colors = ["#FF6347", "#FFA500", "#32CD32"]  # Red, Orange, Green

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
        ax.axis("equal")

        # Display chart
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# ================= CSV/TXT FILE UPLOAD ================= #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>📂 Batch Sentiment Analysis (CSV/TXT Upload)</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        text_data = uploaded_file.read().decode("utf-8")
        df = pd.DataFrame({"review": text_data.split("\n")})

    if "review" not in df.columns:
        st.error("❌ The file must contain a column named 'review' or be a valid TXT file.")
    else:
        df["sentiment_score"] = df["review"].apply(lambda x: predict_sentiment(str(x))[0])
        df["sentiment_label"] = df["sentiment_score"].apply(lambda x: "😠 Negative" if x < 3 else "😐 Neutral" if x == 3 else "😃 Positive")

        # Save all reviews to MongoDB
        for _, row in df.iterrows():
            save_to_db(row["review"], row["sentiment_score"], [])

        # Display Analyzed Data
        st.write("### 📋 Analyzed Reviews", df[["review", "sentiment_label"]])

        # Sentiment Distribution
        sentiment_counts = df["sentiment_label"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=["#FF6347", "#FFA500", "#32CD32"])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

# ================= Display Past Reviews from MongoDB ================= #
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>📊 Past Sentiment Analyses</h2>", unsafe_allow_html=True)

if st.button("Load Past Analyses"):
    past_reviews = fetch_past_reviews()
    if not past_reviews.empty:
        st.dataframe(past_reviews)
    else:
        st.info("No past analyses found.")

