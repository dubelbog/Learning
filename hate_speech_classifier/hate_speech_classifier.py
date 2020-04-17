"""
Hate speech classification baseline using sklearn
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
"""
from keras_preprocessing.sequence import pad_sequences

__author__ = "don.tuggener@zhaw.ch"

import keras
import sys
import random
import pandas as pd
import zipfile
import pickle
from keras_preprocessing.text import Tokenizer

from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict, train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

random.seed(42)  # Ensure reproducible results
STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words('english')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('learning_accuracy_loss.png')
    plt.show()


if __name__ == '__main__':
    print('Loading data', file=sys.stderr)
    zip_ref = zipfile.ZipFile('train.csv.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()
    text2str = ""
    df = pd.read_csv("train.csv")
    df = df[["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

    df = df.sample(frac=1).reset_index(drop=True)

    review = df['comment_text']
    toxic = df["toxic"]
    severe_toxic = df["severe_toxic"]
    obscene = df["obscene"]
    threat = df["threat"]
    insult = df["insult"]
    identity_hate = df["identity_hate"]

    label = pd.concat([toxic, severe_toxic, obscene, threat, insult, identity_hate], axis=1)
    # sum all result in the rows
    label = label.sum(axis=1)
    label = label.to_numpy()

    print("create tokanizer")
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    print("fit tokanizer")
    tokenizer.fit_on_texts(review)
    review_seq = tokenizer.texts_to_sequences(review)
    review_seq_pad = pad_sequences(review_seq)

    # encoded = tokenizer.texts_to_sequences([text2str])
    # word2idx = tokenizer.word_index
    # idx2word = tokenizer.index_word

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Classification and evaluation', file=sys.stderr)

    # Randomly split data into 80% training and 20% testing, preserve class distribution with stratify
    X_train, X_test, Y_train, Y_test = train_test_split(review_seq_pad, insult, test_size=0.20, random_state=42)

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    num_classes = 2
    batch_size = 512
    epochs = 50

    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)

    clf = Sequential()
    input_dim = X_train.shape[1]
    clf.add(layers.Dense(16, input_dim=input_dim, activation='relu'))
    clf.add(layers.Dense(16, input_dim=input_dim, activation='relu'))
    clf.add(layers.Dense(2, input_dim=input_dim, activation='sigmoid'))
    clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    clf.summary()

    history = clf.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=False,
                      validation_data=(X_test, Y_test))

    loss, accuracy = clf.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = clf.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plt.style.use('ggplot')
    plot_history(history)

    clf.save('Sentiment.h5')

    max_review_length = 1401

    # review = ["idiot i hate you", "really liked movie had fun", "movie was terrible bad", "don't like", "love",
    #            "you are stupid and dopey"]

    txt = "really liked movie had fun"
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_review_length)
    pred = clf.predict_classes(padded)
    print(pred)
    print(clf.predict(padded))

    """
    # Apply cross-validation, create prediction for all data point
    numcv = 3   # Number of folds
    print('Using', numcv, 'folds', file=sys.stderr)
    y_pred = cross_val_predict(clf, X_tfidf_matrix, Y, cv=numcv)
    print(classification_report(Y, y_pred), file=sys.stderr)
    """
