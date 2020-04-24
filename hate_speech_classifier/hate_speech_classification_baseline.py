"""
Hate speech classification baseline using sklearn
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
"""

__author__ = "don.tuggener@zhaw.ch"

import sys
import utils_classifier

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import LinearSVC

if __name__ == '__main__':

    print('Loading data', file=sys.stderr)
    X, Y = utils_classifier.read_data_classifier(reprocess=True)

    print('Vectorizing with TFIDF', file=sys.stderr)
    tfidfizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf_matrix = tfidfizer.fit_transform(X)
    print('Data shape:', X_tfidf_matrix.shape)
    do_downsample = True
    if do_downsample:   # Only take 20% of the data
        X_tfidf_matrix, X_, Y, Y_ = train_test_split(X_tfidf_matrix, Y, test_size=0.8, random_state=42, stratify=Y)
        print('Downsampled data shape:', X_tfidf_matrix.shape)

    print('Classification and evaluation', file=sys.stderr)
    clf = LinearSVC(C=1, tol=0.001)    # Weight samples inverse to class imbalance
    # Randomly split data into 80% training and 20% testing, preserve class distribution with stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=0.2, random_state=42, stratify=Y)

    utils_classifier.plot_learning_curve(clf, X_tfidf_matrix, Y,  'Learning Curve Logistic Regression', 'learning_curve_svm.png')

    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(Y_test, y_pred), file=sys.stderr)
    print(confusion_matrix(Y_test, y_pred.tolist()), file=sys.stderr)

    dump(clf, 'sentiment_baseline.joblib')

    # # Apply cross-validation, create prediction for all data point
    # numcv = 3   # Number of folds
    # print('Using', numcv, 'folds', file=sys.stderr)
    # y_pred = cross_val_predict(clf, X_tfidf_matrix, Y, cv=numcv)
    # print(classification_report(Y, y_pred), file=sys.stderr)

