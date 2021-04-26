import random
import json
import pickle

import numpy as np
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from speech_recog import recognize
from text_to_voice import to_speech

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


# Cleaning up the sentences
def clean_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# convert a sentence into a bag of words
def conv_to_bag(sentence):
    sentence_words = clean_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_label(sentence):
    bag_of_words = conv_to_bag(sentence)
    result = model.predict(np.array([bag_of_words]))[0]
    results = [[i, j] for i, j in enumerate(result) if j > 0.6]
    print(results)
    return_list = []
    if len(results) == 0:
        return_list.append({'intent': 'unknown'})
        return return_list

    results.sort(key=lambda x: x[1], reverse=True)
    for i in results:
        return_list.append({'intent': classes[i[0]], 'probability': str(i[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Go! Bot is running!")


def prompt_user(tag):
    prompts_dict = json.loads(open('prompts.json').read())
    for prompt in prompts_dict['prompts']:
        if prompt['tag'] == tag:
            return prompt['responses'][0]


def extract_info(tag):
    message = recognize()
    print("User:", message)
    # Clean the message and then use spacy for entity recognition
    clean_msg = clean_sentences(message)

    spacy_nlp = spacy.load('en_core_web_sm')
    document = spacy_nlp((" ".join(clean_msg)).strip())
    # Stores all the entities in the sentence such as names or places
    entities = []
    print("There are", len(document.ents), "entities")
    print([(X.text, X.label_) for X in document.ents])
    if tag == "user_name":
        for i in document.ents:
            if i.label_ in ["PERSON"]:
                entities.append(i.text)
                print("Here we are")
                print(i)
    return entities


def start_bot():
    message = recognize()
    ints = predict_label(message)
    res = to_speech(get_response(ints, intents))
    print("User:", message)
    print("Wellbot:", res)
    print("\n")
    return message


if __name__ == '__main__':
    intro = prompt_user("welcome")
    print("Wellbot:", to_speech(intro))
    name = prompt_user("user_name") + " " + extract_info("user_name")[0]
    print("Wellbot:", to_speech(name))
    while True:
        start_bot()
