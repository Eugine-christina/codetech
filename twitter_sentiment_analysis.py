# ==============================
# SENTIMENT ANALYSIS USING TF-IDF + LOGISTIC REGRESSION
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# STEP 1: LOAD DATA
# ==============================
print("📂 Loading dataset...")

train_path = "twitter_training.csv"
val_path = "twitter_validation.csv"

# Load both CSVs (no headers)
train_df = pd.read_csv(train_path, header=None)
val_df = pd.read_csv(val_path, header=None)

# Combine
df = pd.concat([train_df, val_df], ignore_index=True)
print(f"✅ Combined dataset shape: {df.shape}")

# Show a few rows
print(df.head())

# ==============================
# STEP 2: ASSIGN COLUMN NAMES
# ==============================
# Some Kaggle datasets have 4 or 8 columns. We'll handle both.

if df.shape[1] >= 4:
    # For 4 or more columns, we assume format: id, tweet_id, sentiment, tweet
    df = df.rename(columns={2: "sentiment", 3: "tweet"})
else:
    raise ValueError("Unexpected CSV format! Please check the dataset.")

# Keep only relevant columns
df = df[["sentiment", "tweet"]]

# ==============================
# STEP 3: CLEAN DATA
# ==============================
print("🧹 Cleaning data...")
df = df.dropna(subset=["sentiment", "tweet"])  # remove missing rows
df = df[df["tweet"].astype(str).str.strip() != ""]  # remove blank tweets

print("✅ After cleaning:", df.shape)
print("Sample rows:")
print(df.head(5))

# ==============================
# STEP 4: ENCODE SENTIMENT
# ==============================
print("🔤 Unique sentiments:", df["sentiment"].unique())

# Some datasets use "Positive", "Negative", "Neutral", "Irrelevant"
# We’ll keep only Positive/Negative/Neutral
df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]

# ==============================
# STEP 5: SPLIT DATA
# ==============================
print("✂️ Splitting train/test...")
X = df["tweet"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("✅ Train size:", len(X_train), "Test size:", len(X_test))

# ==============================
# STEP 6: TF-IDF VECTORIZATION
# ==============================
print("🧠 Performing TF-IDF vectorization...")

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("✅ TF-IDF complete. Shape:", X_train_tfidf.shape)

# ==============================
# STEP 7: LOGISTIC REGRESSION MODEL
# ==============================
print("⚙️ Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)

# ==============================
# STEP 8: EVALUATE MODEL
# ==============================
print("📈 Evaluating model...")
y_pred = clf.predict(X_test_tfidf)

print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# STEP 9: SAMPLE PREDICTIONS
# ==============================
print("\n🎯 Example Predictions:")
sample_texts = [
    "I love this product! It's amazing.",
    "This is the worst thing ever!",
    "It was okay, not great but not bad."
]
sample_vec = vectorizer.transform(sample_texts)
sample_preds = clf.predict(sample_vec)

for text, pred in zip(sample_texts, sample_preds):
    print(f"Tweet: {text}\nPredicted Sentiment: {pred}\n")
# ==============================
# STEP 10: USER INPUT PREDICTION
# ==============================
print("💬 Now you can test your own tweet sentiment!")
while True:
    user_input = input("\nEnter a tweet (or type 'exit' to quit): ").strip()
    if user_input.lower() == "exit":
        print("👋 Exiting sentiment predictor.")
        break
    elif user_input == "":
        print("⚠️ Please enter some text.")
        continue

    # Transform using same TF-IDF vectorizer
    user_tfidf = vectorizer.transform([user_input])
    
    # Predict sentiment
    prediction = clf.predict(user_tfidf)[0]
    print(f"🧠 Predicted Sentiment: {prediction}")
