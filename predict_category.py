import joblib
from pathlib import Path

# Paths
MODELS_PATH = Path("models")
MODEL_FILE = MODELS_PATH / "product_category_model.pkl"
VECTORIZER_FILE = MODELS_PATH / "tfidf_vectorizer.pkl"

def main():
    # Load model and vectorizer
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    print("Product Category Prediction")
    print("Type 'exit' to stop.\n")

    while True:
        product_title = input("Enter product title: ")

        if product_title.lower() == "exit":
            print("Exiting program.")
            break

        X_vec = vectorizer.transform([product_title])
        prediction = model.predict(X_vec)[0]

        print(f"Predicted category: {prediction}\n")

if __name__ == "__main__":
    main()
