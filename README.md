# Sentiment-Polarity-Classification-Using-Logistic-Regression
Overview:

This project aims to create a sentiment polarity classifier using logistic regression. The classifier is trained on movie reviews and predicts whether the review is positive or negative based on its text.

Dataset:

The dataset used for training, validation, and testing is the RT Polarity Dataset. It contains 5,331 positive and 5,331 negative reviews. The data is split as follows:

Training Set: 4,000 positive and 4,000 negative reviews
Validation Set: 500 positive and 500 negative reviews
Test Set: 831 positive and 831 negative reviews



Steps:
Data Loading: The reviews are loaded from rt-polarity.pos and rt-polarity.neg files.
Data Splitting: The dataset is split into training, validation, and test sets.
Feature Extraction: TF-IDF is used to convert text into numerical features.
Model Training: Logistic Regression is trained using the training data.
Model Evaluation: The model is evaluated using the test set. Confusion matrix and classification report (precision, recall, F1-score) are generated.
Visualization: Confusion matrix is visualized using a heatmap.



Dependencies:
Python 3.x
pandas
scikit-learn
matplotlib
seaborn
Usage Instructions:

Install the required dependencies:
pip install pandas scikit-learn matplotlib seaborn
Run the script to train the model and generate the output

python sentiment_analysis.py
The confusion matrix and classification report will be printed. The confusion matrix is also saved as an image confusion_matrix.png.
Output:

Confusion Matrix (Heatmap saved as confusion_matrix.png)
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix for React Component Usage
