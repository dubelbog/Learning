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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':

    print('Loading data', file=sys.stderr)
    X, Y = utils_classifier.read_data_classifier(reprocess=True)

    print('Vectorizing with TFIDF', file=sys.stderr)
    tfidfizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf_matrix = tfidfizer.fit_transform(X)
    print('Data shape:', X_tfidf_matrix.shape)
    do_downsample = True
    if do_downsample:  # Only take 20% of the data
        X_tfidf_matrix, X_, Y, Y_ = train_test_split(X_tfidf_matrix, Y, test_size=0.8, random_state=42, stratify=Y)
        print('Downsampled data shape:', X_tfidf_matrix.shape)

    # Set the parameters by cross-validation
    tuned_parameters = [{'tol': [1e-3, 1e-4], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['saga'], 'penalty': ['l1']},
                        {'tol': [1e-3, 1e-4], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['newton-cg', 'lbfgs']},
                        {'tol': [1e-3, 1e-4], 'C': [0.25, 0.5, 0.75, 1], 'solver': ['newton-cg'], 'penalty': ['none']}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            LogisticRegression(), tuned_parameters, scoring='%s_macro' % score
        )

        X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=0.2, random_state=42, stratify=Y)

        clf.fit(X_train, Y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = Y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

