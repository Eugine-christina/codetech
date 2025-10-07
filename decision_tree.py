# ============================
# Breast Cancer Decision Tree - VS Code Ready
# ============================

# ============================
# STEP 1: Import Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os

# Set plot aesthetics
sns.set(style='whitegrid', palette='muted', font_scale=1.1)

# ============================
# STEP 2: Load Dataset
# ============================
# Set the full path to your CSV file
data_path = r"C:\Users\santh\OneDrive\Desktop\code tech\breast_cancer.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"CSV file not found at {data_path}. Please check the path!")

df = pd.read_csv(data_path)
print("Dataset loaded successfully!")
print(df.head())

# ============================
# STEP 3: Basic Info & EDA
# ============================
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['diagnosis'].value_counts())

# Plot class distribution
sns.countplot(x='diagnosis', data=df)
plt.title("Diagnosis Counts")
plt.show()

# ============================
# STEP 4: Preprocessing
# ============================
df = df.drop(['id','Unnamed: 32'], axis=1, errors='ignore')
df['target'] = df['diagnosis'].map({'M':1, 'B':0})
df = df.drop('diagnosis', axis=1)

print("\nPreprocessed DataFrame:")
print(df.head())

# ============================
# STEP 5: Feature Correlation & Plots
# ============================
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

top_features = df.corr()['target'].abs().sort_values(ascending=False).index[1:6]
df[top_features].hist(figsize=(12,8))
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='target', y='radius_mean', data=df)
plt.title("Radius Mean by Target")
plt.show()

sns.pairplot(df[list(top_features) + ['target']], hue='target')
plt.show()

# ============================
# STEP 6: Train-Test Split
# ============================
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ============================
# STEP 7: Train Basic Decision Tree
# ============================
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# ============================
# STEP 8: Evaluate Basic Tree
# ============================
y_pred = clf.predict(X_test)

y_proba = clf.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("ROC AUC:", roc_auc_score(y_test, y_proba))

# ============================
# STEP 9: Visualize Basic Tree
# ============================
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['Benign','Malignant'], filled=True, rounded=True, max_depth=3)
plt.show()

print(export_text(clf, feature_names=list(X.columns))[:1500])

feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()

# ============================
# STEP 10: Hyperparameter Tuning (GridSearchCV)
# ============================
param_grid = {
    'max_depth': [3, 4, 5, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

grid.fit(X_train, y_train)

print("Best hyperparameters:", grid.best_params_)

best_clf = grid.best_estimator_

y_pred_best = best_clf.predict(X_test)
y_proba_best = best_clf.predict_proba(X_test)[:,1]

print("Accuracy (best):", accuracy_score(y_test, y_pred_best))
print("ROC AUC (best):", roc_auc_score(y_test, y_proba_best))
print("\nClassification Report (best):\n", classification_report(y_test, y_pred_best))

# ============================
# STEP 11: Cost-Complexity Pruning
# ============================
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

clfs = []
for alpha in ccp_alphas:
    dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    dt.fit(X_train, y_train)
    clfs.append(dt)

test_scores = [clf.score(X_test, y_test) for clf in clfs]

plt.figure(figsize=(10,5))
plt.plot(ccp_alphas, test_scores, marker='o')
plt.xlabel("ccp_alpha")
plt.ylabel("Test Accuracy")

plt.title("Pruning: ccp_alpha vs Test Accuracy")
plt.show()

best_alpha_index = test_scores.index(max(test_scores))
best_alpha = ccp_alphas[best_alpha_index]
print("Best ccp_alpha:", best_alpha)

pruned_clf = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
pruned_clf.fit(X_train, y_train)

y_pred_pruned = pruned_clf.predict(X_test)
y_proba_pruned = pruned_clf.predict_proba(X_test)[:,1]

print("Accuracy (pruned):", accuracy_score(y_test, y_pred_pruned))
print("ROC AUC (pruned):", roc_auc_score(y_test, y_proba_pruned))

# ============================
# STEP 12: Visualize Pruned Tree
# ============================
plt.figure(figsize=(20,10))
plot_tree(pruned_clf, feature_names=X.columns, class_names=['Benign','Malignant'], filled=True, rounded=True, max_depth=3)
plt.show()

feat_imp_pruned = pd.Series(pruned_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp_pruned.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances (Pruned Tree)")
plt.show()

# ============================
# STEP 13: Save Final Model Locally
# ============================
model_save_path = r"C:\Users\santh\OneDrive\Desktop\code tech\breast_cancer_decision_tree_final.joblib"
joblib.dump(pruned_clf, model_save_path)
print("Final model saved at:", model_save_path)
