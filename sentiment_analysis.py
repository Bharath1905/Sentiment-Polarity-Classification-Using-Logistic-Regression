import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
with open("rt-polarity.pos", "r", encoding="latin-1") as file:
    positive_reviews = file.readlines()
with open("rt-polarity.neg", "r", encoding="latin-1") as file:
    negative_reviews = file.readlines()

# Split the data into training, validation, and test sets
train_pos = positive_reviews[:4000]
train_neg = negative_reviews[:4000]
valid_pos = positive_reviews[4000:4500]
valid_neg = negative_reviews[4000:4500]
test_pos = positive_reviews[4500:]
test_neg = negative_reviews[4500:]

# Combine and label the data
train_data = pd.DataFrame({
    "text": train_pos + train_neg,
    "label": [1] * len(train_pos) + [0] * len(train_neg)
})
valid_data = pd.DataFrame({
    "text": valid_pos + valid_neg,
    "label": [1] * len(valid_pos) + [0] * len(valid_neg)
})
test_data = pd.DataFrame({
    "text": test_pos + test_neg,
    "label": [1] * len(test_pos) + [0] * len(test_neg)
})

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train_data["text"])
X_test = tfidf.transform(test_data["text"])
y_train = train_data["label"]
y_test = test_data["label"]

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_test_pred = model.predict(X_test)

# Generate the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

# Print results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.close()

# Print the confusion matrix values for easy copying
print("\nConfusion Matrix Values for React Component:")
print(f"[[{conf_matrix[0][0]}, {conf_matrix[0][1]}],")
print(f" [{conf_matrix[1][0]}, {conf_matrix[1][1]}]]")