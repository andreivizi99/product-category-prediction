import joblib

MODEL_PATH = "models/product_category_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def main():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Product Category Prediction")
    print("Type 'exit' to stop.\n")

    while True:
        title = input("Enter product title: ")

        if title.lower() == "exit":
            break

        title_vec = vectorizer.transform([title])
        prediction = model.predict(title_vec)[0]

        print(f"Predicted category: {prediction}\n")

if __name__ == "__main__":
    main()
