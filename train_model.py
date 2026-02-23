"""
News Article Credibility Classifier — Training Pipeline
========================================================
Trains Logistic Regression & Passive Aggressive classifiers on the WELFake
dataset and serializes the best model + TF-IDF vectorizer to disk.

Labels:  1 = Credible / True   |   0 = Fake / Misinformation
"""

import os
import re
import time
import warnings

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "WELFake_Dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TFIDF_MAX_FEATURES = 50_000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip HTML, remove non-alpha characters."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


# ── 1. Load & Clean ─────────────────────────────────────────────────────────
print("=" * 60)
print("  News Credibility Classifier — Training Pipeline")
print("=" * 60)

print("\n[1/5] Loading dataset …")
df = pd.read_csv(DATASET_PATH)
print(f"      Raw rows: {len(df):,}")

# Drop rows with missing title or text
df.dropna(subset=["title", "text"], inplace=True)
df.drop_duplicates(subset=["title", "text"], inplace=True)
print(f"      After cleanup: {len(df):,}")

# Label distribution
print(f"\n      Label distribution:")
for label, count in df["label"].value_counts().sort_index().items():
    tag = "Fake" if label == 0 else "Credible"
    print(f"        {label} ({tag}): {count:,}")

# Combine title + text → content
df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)


# ── 2. Feature Engineering ───────────────────────────────────────────────────
print("\n[2/5] Building TF-IDF features …")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X = tfidf.fit_transform(df["content"])
y = df["label"]
print(f"      Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")


# ── 3. Train / Test Split ───────────────────────────────────────────────────
print("\n[3/5] Splitting data (80 / 20 stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(f"      Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")


# ── 4. Train Models ─────────────────────────────────────────────────────────
print("\n[4/5] Training models …\n")

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Passive Aggressive": PassiveAggressiveClassifier(
        max_iter=50, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

best_name, best_model, best_f1 = None, None, 0.0

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    cm = confusion_matrix(y_test, preds)

    print(f"  ▸ {name}")
    print(f"    Trained in {elapsed:.1f}s")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1 Score : {f1:.4f}")
    print(f"    Confusion Matrix:")
    print(f"      {cm[0]}")
    print(f"      {cm[1]}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, preds, target_names=["Fake (0)", "Credible (1)"]))
    print("-" * 50)

    if f1 > best_f1:
        best_name, best_model, best_f1 = name, model, f1


# ── 5. Serialize ─────────────────────────────────────────────────────────────
print(f"\n[5/5] Best model: {best_name} (F1={best_f1:.4f})")
print(f"      Saving to {MODEL_DIR}/ …")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))

# Save metadata
with open(os.path.join(MODEL_DIR, "metadata.txt"), "w") as f:
    f.write(f"model_name: {best_name}\n")
    f.write(f"f1_score: {best_f1:.4f}\n")
    f.write(f"tfidf_max_features: {TFIDF_MAX_FEATURES}\n")
    f.write(f"train_samples: {X_train.shape[0]}\n")
    f.write(f"test_samples: {X_test.shape[0]}\n")

print("\n✅  Training complete. Model ready for serving.\n")
