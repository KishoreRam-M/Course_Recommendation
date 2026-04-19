import time
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

from config.settings import (
    SEED, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
    KNN_NEIGHBORS, KNN_METRIC, KNN_ALGORITHM,
    BEST_MODEL_PATH, VECTORIZER_PATH, LABEL_ENC_PATH, RECOMMENDER_PATH, METADATA_PATH
)

def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=5, max_iter=1000, solver='lbfgs', multi_class='auto', n_jobs=-1, random_state=SEED
        ),
        "Linear SVM": LinearSVC(
            C=1.0, max_iter=2000, random_state=SEED
        ),
        "Multinomial NaiveBayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=15, n_jobs=-1, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=10, learning_rate=0.1, max_depth=3,
            subsample=0.8, random_state=SEED
        ),
    }

def train_and_evaluate(df: pd.DataFrame):
    print("🔄 Label Encoding target...")
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    
    print("🔄 Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    X = vectorizer.fit_transform(df['clean_text'])
    
    print(f"   ✅ TF-IDF matrix: {X.shape[0]:,} × {X.shape[1]:,}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    models = get_candidate_models()
    results_log = []
    trained_models = {}

    print("\n🚀 Starting Model Training...")
    
    for name, clf in models.items():
        print(f"⏳ Training {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        trained_models[name] = clf
        results_log.append({
            "Model": name,
            "Accuracy": round(acc * 100, 2),
            "F1-Score (weighted)": round(f1 * 100, 2),
            "Train Time (s)": round(elapsed, 1),
        })
        print(f"  ✅ Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%\n")

    best_record = max(results_log, key=lambda x: x['F1-Score (weighted)'])
    best_name = best_record["Model"]
    best_model = trained_models[best_name]

    print(f"🏆 Best Model Selected: {best_name} (F1: {best_record['F1-Score (weighted)']}%)")

    save_artifacts(best_model, vectorizer, le, df, X, best_name, results_log, X_train.shape[0], X_test.shape[0])

def save_artifacts(best_model, vectorizer, le, df, X_full, best_name, results_log, n_train, n_test):
    print("\n🔄 Saving models and metadata...")
    joblib.dump(best_model, BEST_MODEL_PATH, compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    joblib.dump(le, LABEL_ENC_PATH, compress=3)
    
    print("🔄 Building recommender index (k-NN)...")
    knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, metric=KNN_METRIC, algorithm=KNN_ALGORITHM, n_jobs=-1)
    knn.fit(X_full)
    joblib.dump(knn, RECOMMENDER_PATH, compress=3)

    meta = {
        "best_model_name": best_name,
        "classes": list(le.classes_),
        "n_features": vectorizer.max_features,
        "n_train": n_train,
        "n_test": n_test,
        "leaderboard": results_log,
        "columns": df.columns.tolist(),
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print("✅ All artifacts saved successfully!")
