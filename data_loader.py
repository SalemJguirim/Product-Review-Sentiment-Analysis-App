import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

def load_dataset():
    df = pd.read_csv("amazon_reviews.csv")
    df = df[['reviewText', 'overall']].dropna()
    df['label'] = df['overall'].apply(lambda x: int(x) - 1)
    return df

class SentimentDataset(Dataset):
    def __init__(self, reviews, labels):
        from utils import preprocess_text
        self.reviews = reviews
        self.labels = labels
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text = self.reviews[idx]
        label = self.labels[idx]
        tokens = preprocess_text(text)
        return {**tokens, 'label': torch.tensor(label)}