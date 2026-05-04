# SMS Spam Classifier

An NLP pipeline that classifies SMS messages as **spam** or **ham (not spam)** using TF-IDF vectorization and Multinomial Naive Bayes — with a live Streamlit interface.

---

## Problem

SMS inboxes are flooded with promotional messages, phishing links, and scams. This classifier detects spam in real time using classical NLP techniques — lightweight, interpretable, and fast.

---

## How It Works

```
Raw SMS → Preprocess → TF-IDF Vectorize → Naive Bayes → Spam / Not Spam
```

1. **Preprocessing** — Lowercase → tokenize → remove non-alphanumeric tokens → remove stopwords → Porter stemming
2. **Vectorization** — TF-IDF converts the cleaned text into a numeric feature vector
3. **Classification** — Multinomial Naive Bayes predicts spam (1) or ham (0)
4. **Interface** — Streamlit web app for instant predictions

---

## Tech Stack

| Component     | Tool                              |
|---------------|-----------------------------------|
| Language      | Python 3                          |
| NLP           | NLTK (tokenization, stopwords, stemming) |
| Vectorizer    | TF-IDF — `scikit-learn`           |
| Classifier    | Multinomial Naive Bayes — `scikit-learn` |
| App           | Streamlit                         |
| Deployment    | Heroku                            |

---

## Run Locally

```bash
# Clone the repo
git clone https://github.com/huzefa10/sms-spam-classifier.git
cd sms-spam-classifier

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
sms-spam-classifier/
├── app.py              # Streamlit app — preprocessing + prediction UI
├── model.pkl           # Trained Multinomial Naive Bayes model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── nltk.txt            # NLTK packages to download on Heroku
├── Procfile            # Heroku process config
└── setup.sh            # Heroku pre-run setup script
```

---

## Screenshot

*Demo screenshot — add after deployment*

---

## Future Improvements

- Compare with BERT/RoBERTa for accuracy benchmarking
- Show prediction confidence score
- Add CSV batch prediction mode
- Multilingual spam detection
