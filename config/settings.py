from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "courses_en.csv"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Standard Saved Artifact Paths
BEST_MODEL_PATH = MODEL_DIR / "best_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LABEL_ENC_PATH = MODEL_DIR / "label_encoder.pkl"
RECOMMENDER_PATH = MODEL_DIR / "recommender_index.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# General Config
SEED = 42

# TF-IDF Configuration
TFIDF_MAX_FEATURES = 25000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

# KNN Configuration
KNN_NEIGHBORS = 6
KNN_METRIC = 'cosine'
KNN_ALGORITHM = 'brute'
