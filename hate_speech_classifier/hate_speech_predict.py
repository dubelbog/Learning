import pickle

from keras.models import load_model

model = load_model('Sentiment.h5')

with open('vectorizer.pk', 'rb') as handle:
    tokenizer = pickle.load(handle)

review = ["I don’t get why negroes always traveling to white countries. Take your ass to Africa!",
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
          "You are my sunshine and I can not live without you"
          ]

# txt = "Calling you an idiot would be an insult to all the stupid people."

file = open("hate_speech_clf_examples.txt", 'r+')
file.truncate(0)

for txt in review:
    file = open("hate_speech_clf_examples.txt", "a")
    Y = tokenizer.transform([txt])
    prediction = model.predict(Y)
    print(txt)
    print(prediction)

    if abs(prediction[0][0] - prediction[0][1]) < 0.15:
        result = "It is not clear if it's hate comment"
    elif prediction[0][0] < prediction[0][1]:
        result = "It is a hate comments"
    else:
        result = "It is NOT a hate comments"
    print(result)
    file.write(txt + " \n" + "not hate " + str(prediction[0][0]) + " hate " + str(prediction[0][1]) + " \n" + result)
    file.write("\n")
    file.write("\n")
    file.close()

