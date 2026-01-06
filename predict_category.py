import pickle

# Load model
with open("models/product_category_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

print("Interactive product category prediction")
print("Type 'exit' to stop")

while True:
    title = input("\nEnter product title: ")

    if title.lower() == "exit":
        break

    title = title.lower()
    title_tfidf = tfidf.transform([title])
    prediction = model.predict(title_tfidf)

    print("Predicted category:", prediction[0])
