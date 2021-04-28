import random
import json
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# pip3 install --user --upgrade tensorflow
# Tensorflow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


def extract_wrds_and_classes(intents_dict):
    words = []
    classes = []
    documents = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for intent in intents_dict['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in stop_words]
    # Store sorted words and classes
    words = sorted(set(words))

    classes = sorted(set(classes))

    # Store the words in a pickle file
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    return [words, classes, documents]


def train_bow(wrds, clss, docs):
    lemmatizer = WordNetLemmatizer()
    # Training the model
    training = []
    output_empty = [0] * len(clss)

    for document in docs:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in wrds:
            bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[clss.index(document[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    # Features and their labels
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Lets make model bois
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # Allows us to add up results/ scale results to add up to 1
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Define stochstic gradient descent optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.model')
    print('Model Trained')


if __name__ == '__main__':
    intents = json.loads(open('intents.json').read())
    extracted_wrds_and_classes = extract_wrds_and_classes(intents)
    # Train the BOW using the words, classes, and the word_list + tag/docs
    train_bow(extracted_wrds_and_classes[0], extracted_wrds_and_classes[1],
              extracted_wrds_and_classes[2])
