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
    print("Type 'exit' to quit.\n")

    while True:
        title = input("Enter product title: ")

        if title.lower() == "exit":
            print("Goodbye!")
            break

        title_vec = vectorizer.transform([title])
        prediction = model.predict(title_vec)

        print(f"Predicted category: {prediction[0]}\n")

if __name__ == "__main__":
    main()
