import pandas as pd

# Load the CSV files
train_df = pd.read_csv("twitter_training.csv")
val_df = pd.read_csv("twitter_validation.csv")

# Combine datasets
df = pd.concat([train_df, val_df], ignore_index=True)

# Save as one CSV
df.to_csv("twitter_sentiment.csv", index=False)
print("Combined CSV created with shape:", df.shape)
