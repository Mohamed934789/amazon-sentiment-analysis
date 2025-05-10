# amazon-sentiment-analysis
# Amazon Reviews Sentiment Analysis 🛒🔍

This project analyzes customer reviews from Amazon using Natural Language Processing (NLP) techniques to classify them as **positive**, **negative**, or **neutral** using a Machine Learning model.

---

## 📁 Dataset

The dataset contains Amazon product reviews with:
- `Text` reviews
- `Score` (1 to 5)
- Other metadata

We mainly used:
- `reviewText` ➜ the written customer review
- `Score` ➜ the actual rating (target label)

---

## 🧠 Project Goals

- Clean and preprocess the reviews
- Use VADER for sentiment scoring
- Train a **Logistic Regression** model on processed data
- Evaluate model accuracy

---

## ⚙️ Tools and Libraries Used

- **pandas, numpy** – data manipulation
- **matplotlib, seaborn** – data visualization
- **NLTK, TextBlob, VADER** – for text preprocessing and sentiment analysis
- **Scikit-learn (sklearn)** – for ML model, pipeline, vectorization, and evaluation

---

## 🔄 Data Preprocessing Steps

1. **Clean the text**:
   - Remove punctuation
   - Remove stopwords
   - Stemming with `PorterStemmer`

2. **VADER Sentiment Analysis**:
   - Add a `polarity_score` to each review using `SentimentIntensityAnalyzer`

3. **Preprocess Text**:
   - Create a new column: `review_stemmed` with cleaned/stemmed version of the review

---

## 🧪 Model Training

- **Features used**: `review_stemmed`
- **Target label**: `Score`
- Used `CountVectorizer` ➜ then `TF-IDF` ➜ then `StandardScaler`
- Trained with **Logistic Regression**

Used `train_test_split` to split the data (80% train - 20% test)

---

## 📊 Model Evaluation

- Accuracy is printed for both:
  - **Training set**
  - **Test set**
- You can also use `classification_report()` for more metrics (precision, recall, f1-score)

---

## 📈 Future Improvements

- Use other models: Random Forest, XGBoost
- Try `Lemmatization` instead of Stemming
- Add Hyperparameter tuning
- Improve class balancing if needed

---

## ✅ How to Run

1. Clone this repo
2. Make sure to install requirements:
```bash
pip install -r requirements.txt
