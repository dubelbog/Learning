import pickle
import utils_classifier

from keras.models import load_model

model = load_model('Sentiment.h5')

with open('vectorizer.pk', 'rb') as handle:
    tokenizer = pickle.load(handle)

review = utils_classifier.get_reviews()
file_name = "hate_speech_predict_keras_6.txt"
file_name_2 = "hate_speech_predict_keras.txt"

file = open(file_name, 'w+')
file = open(file_name_2, 'w+')
file.truncate(0)

#utils_classifier.analyze_review_6_classes(review, tokenizer, model, file_name)
utils_classifier.analyze_review(review, tokenizer, model, file_name_2)

