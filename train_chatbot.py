import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle
from autocorrect import AutoCorrect
import spacy

nlp = spacy.load('el_core_news_lg')


def train_pipeline():
    # ---------------------- Load training data
    with open('Data/intents.json', encoding='utf-8') as f:
        training_data = json.load(f)

    # ---------------------- Prepare training data and save what will be needed by the chatbot
    ac = AutoCorrect()
    
    def lemmatize(text):
        # autocorrect and tokenize each word in the text ant then return the token's lemma
        return [tok.lemma_ for tok in nlp(ac.autocorrect(pattern))]
    
    tags = [intent['tag'] for intent in training_data['intents']]
    all_lemmas = []
    for intent in training_data['intents']:
        for pattern in intent['patterns']:
            all_lemmas.extend(lemmatize(pattern))
    all_lemmas = list(set(all_lemmas))

    pickle.dump(all_lemmas, open('Data/all_lemmas_from_patterns.pkl', 'wb'))

    # ---------------------- 
    def bag_of_words(text):
        text_lemmas = lemmatize(text)
        bag = [0] * len(all_lemmas)  # bag of words
        for lem in text_lemmas:
            if lem in all_lemmas:
                bag[all_lemmas.index(lem)] += 1
        return bag

    input = []
    output = []
    for intent in training_data['intents']:
        for pattern in intent['patterns']:
            # create our input bag of words
            input.append(bag_of_words(pattern))
    
            # output is a '0' for each tag and '1' for current tag
            output_row = [0] * len(tags)
            output_row[tags.index(intent['tag'])] = 1
            output.append(output_row)

    # --------------------------- Train the model
    print("Start training")
    print(len(input))
    model = Sequential()
    model.add(Dense(128, input_shape=(len(all_lemmas),), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(tags), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy', 'Precision', 'Recall'])

    # fitting and saving the model
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=5)
    hist = model.fit(np.array(input), np.array(output), epochs=100, batch_size=64, verbose=True, callbacks=[es])
    model.save('Models/intent_classification.h5', hist)

    print("Model trained")


if __name__ == "__main__":
    train_pipeline()
