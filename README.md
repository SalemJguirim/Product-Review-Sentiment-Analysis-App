🛍️ Product Review Sentiment Analysis App (with BERT and MongoDB)
This project is a Streamlit web app that performs sentiment analysis on product reviews using a pre-trained BERT model (nlptown/bert-base-multilingual-uncased-sentiment). It supports both single text input and batch uploads via CSV/TXT, and stores analysis results in MongoDB.

📌 Features
Predicts review sentiment on a scale of 1 to 5 using BERT

Classifies reviews as Negative 😠, Neutral 😐, or Positive 😃

Accepts:

Single review via text input

Batch review files (.csv or .txt)

Saves each analysis result in MongoDB for later use

Displays pie charts and bar graphs for sentiment distribution

🧰 Tech Stack
Model: nlptown/bert-base-multilingual-uncased-sentiment

Frontend: Streamlit

Backend: PyTorch, Transformers

Database: MongoDB 
