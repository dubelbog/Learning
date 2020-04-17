import pickle
import utils_classifier
from joblib import load

model = load('sentiment_baseline.joblib')

with open('vectorizer.pk', 'rb') as handle:
    tokenizer = pickle.load(handle)

review = utils_classifier.get_reviews()
file_name = "hate_speech_predict_baseline.txt"

file = open(file_name, 'r+')
file.truncate(0)

utils_classifier.analyze_review_scikit(review, tokenizer, model, file_name)

