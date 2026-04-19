import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config.settings import DATA_PATH

def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, stop_words = setup_nltk()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)

def load_and_preprocess_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    
    df['combined_text'] = (
        df['name'].fillna('') + ' ' +
        df['skills'].fillna('') + ' ' +
        df['what_you_learn'].fillna('') + ' ' +
        df['content'].fillna('')
    )
    
    df['clean_text'] = df['combined_text'].apply(clean_text)
    return df
