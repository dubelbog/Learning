import pickle
import utils_classifier

from keras.models import load_model

model = load_model('Sentiment.h5')

with open('vectorizer.pk', 'rb') as handle:
    tokenizer = pickle.load(handle)

review = utils_classifier.get_reviews()
file_name = "hate_speech_predict_keras.txt"

file = open(file_name, 'w+')
file.truncate(0)

utils_classifier.analyze_review(review, tokenizer, model, file_name)
