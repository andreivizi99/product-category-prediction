import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path

# Paths
DATA_PATH = Path("data/IMLP4_TASK_03-products (3).csv")
MODELS_PATH = Path("models")
MODELS_PATH.mkdir(exist_ok=True)

MODEL_FILE = MODELS_PATH / "product_category_model.pkl"
VECTORIZER_FILE = MODELS_PATH / "tfidf_vectorizer.pkl"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Keep only required columns
    df = df[["Product Title", "Category Label"]]

    # Drop missing values
    df = df.dropna()

    X = df["Product Title"]
    y = df["Category Label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorization
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
