# Text to voice module for welbot
import pyttsx3


def to_speech(response):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 145)
    engine.setProperty('voice', voices[0].id)
    engine.say(response)
    engine.runAndWait()
    return response
