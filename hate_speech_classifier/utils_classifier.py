import csv
import re
import sys
import pickle
import random
import zipfile


from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

random.seed(42)  # Ensure reproducible results
STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words('english')


def read_data_classifier(remove_stopwords=True, remove_numbers=True, do_stem=True, reprocess=False):
    """
    Read CSV with annotated data.
    We'll binarize the classification, i.e. subsume all hate speach related classes
    'toxic, severe_toxic, obscene, threat, insult, identity_hate'
    into one.

    In this method we also do a lot of preprocessing steps, based on the flags which are set in the parameters.
    Feel free to try out different possible combinations of preprocessing steps (e.g. with cross-validation).
    """
    if reprocess:
        X, Y = [], []
        zip_ref = zipfile.ZipFile('train.csv.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
        for i, row in enumerate(csv.reader(open('train.csv', encoding='UTF-8'))):
            if i > 0:   # Skip the header line
                sys.stderr.write('\r'+str(i))
                sys.stderr.flush()
                text = re.findall('\w+', row[1].lower())
                if remove_stopwords:
                    text = [w for w in text if not w in STOPWORDS]
                if remove_numbers:
                    text = [w for w in text if not re.sub('\'\.,','',w).isdigit()]
                if do_stem:
                    text = [STEMMER.stem(w) for w in text]
                label = 1 if '1' in row[2:] else 0  # Any hate speach label
                X.append(' '.join(text))
                Y.append(label)
        sys.stderr.write('\n')
        pickle.dump(X, open('X.pkl', 'wb'))
        pickle.dump(Y, open('Y.pkl', 'wb'))
    else:
        X = pickle.load(open('X.pkl', 'rb'))
        Y = pickle.load(open('Y.pkl', 'rb'))
    print(len(X), 'data points read')
    print('Label distribution:',Counter(Y))
    print('As percentages:')
    for label, count_ in Counter(Y).items():
        print(label, ':', round(100*(count_/len(X)), 2))
    return X, Y


def plot_learning_curve(clf, X_tfidf_matrix, Y, title, file_name):
    train_sizes, train_scores, valid_scores = learning_curve(clf, X_tfidf_matrix, Y, train_sizes=[1000, 5000, 10000, 15000, 25000], cv=5)
    plt.figure(figsize=(12, 5))
    plt.plot(train_sizes, train_scores, 'b', label='Training scores')
    plt.plot(train_sizes, valid_scores, 'r', label='Validation scores')
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.show()


def get_reviews():
    return ["I don’t get why negroes always traveling to white countries. Take your ass to Africa!",
                     "I don’t get why white girls always traveling to white countries. Go to Africa!",
                     "Only a stupid person considers Google a source of education. "
                     "I get my facts from real life. You could try getting education from a school. " \
                     "You know what a school is don’t you. Its [sic] the place where white people have to pay extra so that " \
                     "‘some people’ can get in for a massive discount. Coconuts like you (brown on outside, white on inside) " \
                     "won’t survive one day among real black people in Africa.",
                     "What a bunch of bull$h!+ from an egoistical liberal idiot. I didn’t vote for Trump, but I voted. " \
                     "So many people who are mad, didn’t. They are world traveling, airbnbers, like this Gen-Y poop, " \
                     "but they’re not registered voters. Yet they sure complain when the person they didn’t vote " \
                     "for doesn’t get elected.",
                     "I miss you. Let’s see each other soon!",
                     "I can’t stop thinking about you.",
                     "Every time I see you I get butterflies in my tummy.",
                     "You are the light in my life.",
                     "Are you always this stupid or are you making a special effort today",
                     "Calling you an idiot would be an insult to all the stupid people.",
                     "You are a sad strange little man, and you have my pity.",
                     "You are my sunshine and I can not live without you",
                     "The world is stupid",
                     "The world is wonderful",
                     "what do you want to know?",
                     "it is an absolute shit"]


def analyze_review(review, tokenizer, model, file_name):
    for txt in review:
        file = open(file_name, "a")
        Y = tokenizer.transform([txt])
        prediction = model.predict(Y)
        print(txt)
        print(prediction)

        if abs(prediction[0][0] - prediction[0][1]) < 0.15:
            result = "It is not clear if it's hate comment"
        elif prediction[0][0] < prediction[0][1]:
            result = "It is a hate comment"
        else:
            result = "It is NOT a hate comment"
        print(result)
        file.write(txt + " \n" + "not hate " + str(prediction[0][0]) + " hate " + str(prediction[0][1]) + " \n" + result)
        file.write("\n")
        file.write("\n")
        file.close()


def analyze_review_6_classes(review, tokenizer, model, file_name):
    for txt in review:
        file = open(file_name, "a")
        Y = tokenizer.transform([txt])
        prediction = model.predict(Y)
        print(txt)
        print(prediction)
        value = 0
        max = 0
        for i in range(7):
            if value < prediction[0][i]:
                value = prediction[0][i]
                max = i

        if max == 0:
            result = "The comment is neutral"
        elif max == 1:
            result = "The comment is toxic"
        elif max == 2:
            result = "The comment is severe_toxic"
        elif max == 3:
            result = "The comment is obscene"
        elif max == 4:
            result = "The comment is threat"
        elif max == 5:
            result = "The comment is insult"
        else:
            result = "The comment is identity_hate"

        print(result)
        file.write(txt + " \n" + "neutral " + str(prediction[0][0]) + " \n" +
                   "toxic " + str(prediction[0][1]) + " \n" +
                   "severe_toxic " + str(prediction[0][2]) + " \n" +
                   "obscene " + str(prediction[0][3]) + " \n" +
                   "threat " + str(prediction[0][4]) + " \n" +
                   "insult " + str(prediction[0][5]) + " \n" +
                   "identity_hate " + str(prediction[0][6]) + " \n" +
                   "result: " + result)
        file.write("\n")
        file.write("\n")
        file.close()


def analyze_review_scikit(review, tokenizer, model, file_name):
    for txt in review:
        file = open(file_name, "a")
        Y = tokenizer.transform([txt])
        prediction = model.predict(Y)
        print(txt)
        print(prediction)

        if prediction[0] == 1:
            result = "It is a hate comment"
        else:
            result = "It is NOT a hate comment"
        print(result)
        file.write(txt + " \n" + str(prediction[0]) + " \n" + result)
        file.write("\n")
        file.write("\n")
        file.close()
