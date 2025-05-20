from transformers import BertTokenizer

# Load tokenizer globally (only once)
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def preprocess_text(text):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
