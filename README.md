# Fake News Detection with Word2Vec and Random Forest

This project trains a **Random Forest** classifier to detect fake news using **Word2Vec** embeddings for feature extraction. The dataset is preprocessed to remove stop words and convert text to vectors, which are then fed into the classifier.

## Files
- `fake_news.csv`: A CSV file containing the dataset. It should include at least two columns:
  - `text`: The text content of the news articles.
  - `label`: The label indicating whether the news is "fake" or "real".

## Dataset

You can download the dataset from this [link](<https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification?resource=download>) and rename it to `fake_news.csv`.

## Prerequisites

To run this project, you'll need:
- A Google Colab environment (recommended).
- The `fake_news.csv` file, which can be uploaded to Google Colab.

## Getting Started

### 1. Upload the Dataset

1. Open Google Colab and create a new notebook.
2. In the left sidebar, click the **Files** icon (folder).
3. Click the **Upload** button and upload your `fake_news.csv` file.
4. After uploading, the file will appear in the Colab file system at `/content/fake_news.csv`.

**Ensure the file is named exactly `fake_news.csv` for the code to work correctly.**

### 2. Install Necessary Libraries

Install the required libraries by running the installation command at the beginning of the notebook.

### 3. Code Overview

This code performs the following steps:
1. **Data Loading and Preprocessing**:
   - Loads `fake_news.csv` and removes rows with missing values.
   - Downloads the NLTK stopwords list and uses it to remove stopwords from the text.
   
2. **Word2Vec Training**:
   - Trains a Word2Vec model on the cleaned text data.
   - Converts each document into an average Word2Vec vector.

3. **Model Training**:
   - Splits the data into training and testing sets.
   - Trains a Random Forest classifier using the Word2Vec vectors as features.

4. **Evaluation**:
   - Predicts on the test data and calculates performance metrics: Accuracy, Precision, Recall, and F1 Score.
   - Visualizes the confusion matrix.

### 4. Running the Code

Copy and paste the code into your Google Colab notebook and run each cell sequentially.

### 5. Results Interpretation

After running the code, you will see:
- **Model Performance Metrics**: Accuracy, Precision, Recall, and F1 Score, which describe the performance of the Random Forest model.
- **Confusion Matrix**: A heatmap showing the model’s performance in predicting "fake" vs "real" labels, indicating any misclassifications.

### 6. Notes

- Ensure the `fake_news.csv` file is in the Colab file system and located at `/content/fake_news.csv`.
- If you encounter any missing library errors, re-run the installation cell.

---
