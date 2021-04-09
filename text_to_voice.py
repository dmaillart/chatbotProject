# Text to voice module for welbot
import pyttsx3


def to_speech(response):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(response)
    engine.runAndWait()
    return response
