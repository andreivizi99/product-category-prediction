import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("data/IMLP4_TASK_03-products (3).csv")

# Clean column names
df.columns = df.columns.str.strip()

# Select relevant columns
df = df[['Product Title', 'Category Label']].dropna()

# Lowercase text
df['Product Title'] = df['Product Title'].astype(str).str.lower()

# Split data
X = df['Product Title']
y = df['Category Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transfor
