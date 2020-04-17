"""
Hate speech classification baseline using sklearn
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
"""

__author__ = "don.tuggener@zhaw.ch"

import keras
import sys
import pickle
import utils_classifier

from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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
    X, Y = utils_classifier.read_data_classifier(reprocess=True)

    print('Vectorizing with TFIDF', file=sys.stderr)
    tfidfizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf_matrix = tfidfizer.fit_transform(X)
    print('Data shape:', X_tfidf_matrix.shape)

    with open('vectorizer.pk', 'wb') as fin:
        pickle.dump(tfidfizer, fin)

    do_downsample = True
    if do_downsample:  # Only take 20% of the data
        X_tfidf_matrix, X_, Y, Y_ = train_test_split(X_tfidf_matrix, Y, test_size=0.8, random_state=42, stratify=Y)
        print('Downsampled data shape:', X_tfidf_matrix.shape)

    print('Classification and evaluation', file=sys.stderr)

    # Randomly split data into 80% training and 20% testing, preserve class distribution with stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=0.2, random_state=42, stratify=Y)

    print('x_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    num_classes = 2
    batch_size = 512
    epochs = 20

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


