# Sentiment Analysis on Reviews 📝🔍

This project focuses on building a sentiment analysis model to classify user reviews as either **positive** or **negative** using Natural Language Processing (NLP) and machine learning techniques.

---

## 📂 Dataset

- **File Used**: `Reviews.csv`
- **Columns**:
  - `Review`: Text review of the product/service.
  - `Liked`: Binary label (1 for positive, 0 for negative).

---

## 🔧 Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Word Cloud and Bar Plot Visualizations
- Tokenization, Stop Words Removal, Stemming, and Lemmatization
- Removal of punctuation, numbers, special characters, emojis, and HTML tags
- TF-IDF Vectorization
- Model Training using **Multinomial Naive Bayes**
- Evaluation using Accuracy and Classification Report
- Custom review prediction with preprocessing pipeline

---

## 🧪 Technologies Used

- Python
- Pandas, Numpy
- Matplotlib, Seaborn
- NLTK, WordCloud, BeautifulSoup, Contractions, Emoji
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository or download the script.
2. Place `Reviews.csv` in the working directory.
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud contractions emoji beautifulsoup4
Download NLTK resources:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
Run the Script
## 🔎 Visualizations
1. Bar plot showing sentiment distribution.

2. Word Cloud of all review content.

3. Frequency plot of keywords like food, place, restaurant.
## 🧼 Preprocessing Steps
- Lowercasing

- Removing punctuation, numbers, and special characters

- Tokenization

- Stopword removal

- Stemming and Lemmatization

- Emoji and HTML tag removal

- Contraction expansion


## 🤖 Model Details
Algorithm: Multinomial Naive Bayes

Vectorization: TF-IDF

Train/Test Split: 80/20

Evaluation Metrics:

Accuracy

Classification Report (Precision, Recall, F1-Score)

## 🧠 Predict New Review
You can enter your own review and get an instant prediction:


Enter a review: "The food was absolutely amazing!"
Output: The review is predicted positive
 ## 📈 Sample Output
 Accuracy: 0.85
Classification Report:
  ```bash
              precision    recall  f1-score   support

           0       0.84      0.87      0.85       203
           1       0.86      0.83      0.85       197

    accuracy                           0.85       400
   macro avg       0.85      0.85      0.85       400
weighted avg       0.85      0.85      0.85       400
 ```bash


##  📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

  ```bash

Let me know if you'd like a separate `preprocess_review()` function definition to include in your script or need a `requirements.txt` file too!
 ```bash
