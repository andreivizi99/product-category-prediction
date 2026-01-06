import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = "data/IMLP4_TASK_03-products (3).csv"
MODEL_PATH = "models/product_category_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df[['Product Title', 'Category Label']].dropna()

    X = df['Product Title']
    y = df['Category Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Training completed. Accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

if __name__ == "__main__":
    main()
