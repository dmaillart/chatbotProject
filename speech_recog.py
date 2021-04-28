# Voice to text module
# Using Speech Recognition
# pip install speech_recognition
import speech_recognition as sr


def recognize():
    # for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
    go_again = True
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        while go_again:
            try:
                print("Say something!")
                audio = r.listen(source)
                msg = r.recognize_google(audio)
                go_again = False
            except:
                error_msg = "Sorry I didn't quite get that, could you repeat yourself please?"
                print(error_msg)
                go_again = True

    return msg
