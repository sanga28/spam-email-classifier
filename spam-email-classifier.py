# spam_email_classifier.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import string

# Download the NLTK stopwords (if you haven't already)
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('spam_data.csv')  # Assuming the dataset has 'text' and 'label' columns
print(df.head())

# Preprocessing the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the text column
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])

# Labels (spam = 1, non-spam = 0)
y = df['label'].map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test with a new email (example)
def predict_spam_or_ham(email):
    email_cleaned = preprocess_text(email)
    email_vec = vectorizer.transform([email_cleaned])
    prediction = clf.predict(email_vec)
    return 'Spam' if prediction == 1 else 'Ham'

# Test the classifier with a new email
new_email = "Congratulations, you've won a free vacation. Click here to claim your prize!"
print(f"Prediction: {predict_spam_or_ham(new_email)}")
