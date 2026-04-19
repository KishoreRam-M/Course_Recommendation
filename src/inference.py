import joblib
import json
import pandas as pd
from typing import Dict, Tuple, List
from pathlib import Path

from config.settings import (
    BEST_MODEL_PATH, VECTORIZER_PATH, LABEL_ENC_PATH, 
    RECOMMENDER_PATH, METADATA_PATH, DATA_PATH
)
from src.data_processing import clean_text

class CourseRecommenderEngine:
    def __init__(self):
        self.clf = None
        self.vec = None
        self.le = None
        self.knn = None
        self.meta = {}
        self.df = pd.DataFrame()
        self.loaded = False
        
    def load_artifacts(self) -> bool:
        try:
            self.clf = joblib.load(BEST_MODEL_PATH)
            self.vec = joblib.load(VECTORIZER_PATH)
            self.le = joblib.load(LABEL_ENC_PATH)
            self.knn = joblib.load(RECOMMENDER_PATH)
            
            with open(METADATA_PATH, "r") as f:
                self.meta = json.load(f)
                
            self.df = pd.read_csv(DATA_PATH)
            self.loaded = True
            return True
        except Exception as e:
            print(f"Failed to load artifacts: {e}")
            return False

    def classify(self, text: str) -> pd.DataFrame:
        if not self.loaded: return pd.DataFrame()
        
        cleaned = clean_text(text)
        v = self.vec.transform([cleaned])
        
        if hasattr(self.clf, 'predict_proba'):
            probs = self.clf.predict_proba(v)[0]
            results = [{"Category": self.le.classes_[i], "Confidence": p * 100} for i, p in enumerate(probs)]
        elif hasattr(self.clf, 'decision_function'):
            scores = self.clf.decision_function(v)[0]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            results = [{"Category": self.le.classes_[i], "Confidence": s * 100} for i, s in enumerate(scores)]
        else:
            pred = self.clf.predict(v)[0]
            results = [{"Category": self.le.classes_[pred], "Confidence": 100.0}]
            
        return pd.DataFrame(results).sort_values(by="Confidence", ascending=False).reset_index(drop=True)

    def recommend(self, text: str, n_results: int = 5) -> List[Dict]:
        if not self.loaded: return []
        
        cleaned = clean_text(text)
        q_vec = self.vec.transform([cleaned])
        dists, idxs = self.knn.kneighbors(q_vec, n_neighbors=n_results + 1)
        
        rows = []
        for dist, idx in zip(dists[0][1:], idxs[0][1:]):
            row = self.df.iloc[idx]
            rows.append({
                "Course Name": row['name'],
                "Category": row['category'],
                "Similarity": f"{(1 - dist) * 100:.1f}%",
                "Skills": ', '.join(str(row.get('skills', '')).split(',')[:4]),
                "URL": row.get('url', '')
            })
        return rows
