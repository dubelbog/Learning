import pickle
import utils_classifier
from joblib import load

model = load('sentiment_regression.sav')

with open('vectorizer.pk', 'rb') as handle:
    tokenizer = pickle.load(handle)

review = utils_classifier.get_reviews()

file_name = "hate_speech_predict_regression.txt"

file = open(file_name, 'w+')
file.truncate(0)

utils_classifier.analyze_review_scikit(review, tokenizer, model, file_name)

