# Voice to text module
# Using Speech Recognition
import speech_recognition as sr


def recognize():
    # for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        print("Say something!")
        audio = r.listen(source)
        msg = r.recognize_google(audio)
    return msg

