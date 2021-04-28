Wellbot: A Chatbot project for CIS4930 - Natural Language Processing at UF

To run the program you must first make sure that you have the necessary dependencies installed. This includes:
  1. NumPy                    [pip install numpy]
  2. nltk                     [pip install nltk]
  3. TensorFlow               [pip install tensorflow]
  4. spaCy                    [pip install spacy]   &&    [python -m spacy download en_core_web_sm]
  5. SpeechRecognition        [pip install SpeechRecognition]
  6. TextToSpeech (pyttsx3)   [pip install pyttsx3]
  
  
Within speech_recog.py, it also may be necessary for the user to update the device_index variable in order for the program to use the microphone.
A list of available microphone devices and their indexes can be found by uncommenting lines 7 & 8 and then running the script.

Once the necessary packages are installed and the correct microphone is identified, to run the program the user should first run the train.py script and let it train the model. Once that finishes, the user will run chatbot.py and begin communicating with Wellbot.
