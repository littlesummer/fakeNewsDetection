# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data and drop rows with NaN values
data = pd.read_csv('WELFake_Dataset.csv')
data.dropna(subset=['text', 'label'], inplace=True)
print("Data loaded and cleaned. Remaining NaN values:")
print(data.isnull().sum())

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        words = [word for word in text.split() if word not in stop_words]
        return ' '.join(words)  # Return a string
    else:
        return ''

# Preprocess the data
print("Starting data preprocessing...")
data['cleaned_text'] = data['text'].apply(preprocess_text)
print("Data preprocessing completed.")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'], data['label'], test_size=0.2, random_state=42
)

# Initialize the CountVectorizer with limited vocabulary
vectorizer = CountVectorizer(max_features=50)

# Fit and transform the training data, transform the test data
print("Transforming text data into TF features with limited vocabulary...")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print("Transformation completed.")

# Train the LinearSVC model
print("Training the LinearSVC model with TF features...")
linear_svc_model = LinearSVC(random_state=42, max_iter=10000)
linear_svc_model.fit(X_train_vect, y_train)
print("Model training completed.")

# Predict and calculate performance metrics
print("Evaluating model performance...")
y_pred = linear_svc_model.predict(X_test_vect)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
f1 = f1_score(y_test, y_pred, pos_label=1.0, average='binary')

print("Model performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Visualization: Confusion Matrix
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# Feature Importance Visualization

# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = linear_svc_model.coef_[0]  # Corrected line

# Create a DataFrame of features and coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by absolute value of coefficients
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)

# Plot the top positive and negative coefficients
plt.figure(figsize=(12, 10))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(50))
plt.title('Top Features Contributing to Fake News Detection')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()