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
