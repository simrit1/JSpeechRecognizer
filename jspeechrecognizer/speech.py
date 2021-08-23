
"""
JSpeechRecognizer pa
"""

import sounddevice as sd
import pvporcupine
import vosk
import time
import json
import speech_recognition
import wave
import numpy as np
import threading
import os

from tensorflow.keras.models import load_model

RATE = 16000
DURATION = 0.5
CHANNELS = 1
CHUNK = 512
MAX_FREQ = 18

sd.default.samplerate = RATE
sd.default.channels = CHANNELS

def _callback(d):
    pass

def formatPredictions(predictions):
    predictions = [[i, float(r)] for i, r in enumerate(predictions)]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

class VoskRecognizer:

    """
    ## Vosk Recognizer Class
    Vosk is an offline speech recognizer that is also able to do recognition in realtime.
    Only downside is if you want fairly accurate results you would have to provide it with one of the larger models (which uses a lot of memory)

    Check the models here: https://alphacephei.com/vosk/models
    You would need to download a model before using this recognizer
    
    ### Parameters
    `*modelPath`: path to your vosk model
    `callback`: a callback function that takes one parameter

    """

    def __init__(self, modelPath: str, callback=_callback):
        self.model = vosk.Model(modelPath)
        self.rec = vosk.KaldiRecognizer(self.model, RATE)
        self.callback = callback
    
    def recognize(self, dataStream: bytes, isSpeech: bool):
        full = {"text": ""}
        partial = {"partial": ""}

        if self.rec.AcceptWaveform(dataStream):
            full = json.loads(self.rec.Result())
        else:
            partial = json.loads(self.rec.PartialResult())
        
        if partial['partial'] and isSpeech:
            return (partial['partial'], True, "partial")
        
        if full['text'] or not isSpeech:
            self.rec.Reset()
            text = full['text']
            if not text:
                text = partial['partial']
            return (text, True, "full")
        return ("", False, "")

class GoogleRecognizer:
    
    """
    ## Google Recognizer Class
    A recognizer that uses google's speech recognition from the speech_recognition library
    And yes it's really accurate because it's made by google, however it requires an internet connection.

    ### Parameters
    `*cachePath`: A .wav file to save the temporary .wav it creates
    `callback`: A callback that takes one parameter

    """

    def __init__(self, cachePath: str, callback=_callback):

        self.callback = callback

        self.recognizer = speech_recognition.Recognizer()
        self.frames = []
        self.cachePath = cachePath
    
    def _save(self):
        wf = wave.open(self.cachePath, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(self.frames))
        wf.close()
    
    def _clear(self):
        self.frames.clear()
    
    def recognize(self, dataStream: bytes, isSpeech: bool):
        if isSpeech:
            self.frames.append(dataStream)
            return ("", False, "")
        else:
            self._save()

            # Convert the cache file into AudioData which the speech_recognition module can utilize
            try:
                with speech_recognition.AudioFile(self.cachePath) as src:
                    audio = self.recognizer.record(src)
            except Exception as e:
                self._clear()
                return ("", False, f"error: {e}")
            
            self._clear()

            try:
                self.callback({"type": "googleRecognizing"})
                content = self.recognizer.recognize_google(audio)
                return (content, True, "full")
            except speech_recognition.UnknownValueError:
                return ("", False, "unrecognized")
            except Exception as e:
                return ("", False, "error: {e}")

class JarvisVAD:

    """
    ## Voice Activity Detection
    This is not meant to be a replacement for webrtcvad but I've found that webrtcvad 
    doesn't seem to filter out voice activity in the background (such as some music playing etc)

    This uses deep learning to predict whether a piece of audio is classified as speech or not
    It does this by checking the decibels of some frequencies and figuring out whether those decibels is considered someone speaking or not

    NOTE: This may not be accurate depending on your situation, this has only been trained under my own environment.
        Feel free to use something like webrtcvad for the voice activity detection
    
    ### Parameters
    `*modelPath`: path to the vad model
    `callback`: A callback function that takes one parameter
    `sensitivity`: How sensitive the prediction is, values below 0.90 don't really matter much but you can give it a try

    You can use this as a standalone or pass it to the `SpeechRecognizer` class which it requires.

    ## Standalone usage
    To start classifying simply pass a bytes stream into the `isSpeech` function of this class.
    
    Below is the required parameters
    `RATE`: 16000
    `CHANNELS`: 1
    `CHUNK`: 512 
    """

    def __init__(self, modelPath: str, callback=_callback, sensitivity: float = 0.90):
        self.model = load_model(modelPath)
        self.callback = callback

        self.buffer = []
        self.sensitivity = sensitivity
    
    def isSpeech(self, stream: bytes):
        
        # Convert to a numpy array and get decibels
        arr = np.frombuffer(stream, dtype=np.int16)
        db = 20*np.log10(np.abs(np.fft.rfft(arr[:2048])))
        
        # Appends the decibel values of the relevant frequencies
        features = list(np.round(db[3:MAX_FREQ], 2))
        self.buffer.append(features)

        if len(self.buffer) == int(RATE/CHUNK*DURATION):
            total = np.array([x for y in self.buffer for x in y])
            self.buffer.clear()

            # Makes a prediction based on decibel values collected in the span of DURATION seconds
            predictions = self.model.predict(np.array([total]))[0]
            predictions = formatPredictions(predictions)

            index, probability = predictions[0]

            if index == 1 and probability >= self.sensitivity:
                return True
        return False

class SpeechRecognizer:

    """
    ## Main Speech Recognizer Class
    This class will be responsible for running your speech recognition software.
    
    ### Parameters
    `*recognizer`: A recognizer object either `GoogleRecognizer` or `VoskRecognizer`. 
                You can go as far as creating your own recognizer object.
    `*vad`: A voice activity detection object, most likely you're going to use `JarvisVAD`
            but again you can go as far as creating your own.
    `callback`: a function that takes one parameter, all the data that `SpeechRecognizer` generates will be sent here.
    ```
    def callback(data):
        print(data)
    ```

    `wakewords`: a `List` of wakewords, read more on [pvporcupine's documentation](https://pypi.org/project/pvporcupine/)

    `sensitivites`: a `List` of `float` values that correspond to each wakeword, [pvporcupine's documentation](https://pypi.org/project/pvporcupine/)
    `speechLength`: a `Float` that dictates how many seconds of silence it has to recognize before categorizing it as complete
                By default this is 0.9 seconds and that may be too short for people who don't speak fast

    """

    def __init__(self, 
                recognizer,
                vad,
                callback = _callback,
                wakewords: list = ["jarvis"], 
                sensitivities: list = [0.8], 
                speechLength: float = 0.9
        ):

        self.recognizer = recognizer
        self.callback = callback
        self.wakewords = wakewords
        self.sensitivities = sensitivities
        self.vad = vad

        if self.recognizer.callback == _callback:
            self.recognizer.callback = self.callback
        
        self.speechLength = speechLength
        self.realSpeechLength = speechLength
        self.startSpeechLength = 4

        self.woke = False
        self.listen = True

        self.porcupine = pvporcupine.create(keywords=self.wakewords, sensitivities=self.sensitivities)

        self._prevSpeaking = None
        self._speakingLength = 0
        self._frames = []
        self._count = 0
        self._wakeWordStreams = []
        self._isSpeech = True
    
    def _reset(self):
        self.woke = False
        self._count = 0
        self._prevSpeaking = time.time()
        self._speakingLength = 0
        self._isSpeech = True
    
    def _recognize(self, data, frames, t, st):
        arr = np.frombuffer(data, dtype=np.int16)
        wakeWordIndex = self.porcupine.process(arr)

        # This stores the wakeword streams so that the words "hey (wakeword)" is included
        if not self.woke:
            if len(self._wakeWordStreams) >= 30:
                self._wakeWordStreams.pop(0)
            self._wakeWordStreams.append(data)

        if wakeWordIndex >= 0 and not self.woke:
            self.woke = True
            self._count = 0
            self._prevSpeaking = time.time()
            
            self.callback({"type": "wakeWordDetected"})
        
        if self.woke:

            # Collect raw streams used for saving the file (if you want to)
            self._frames.append(bytes(data))

            # Perform voice activity detection
            vad = self.vad.isSpeech(data)
            if vad:
                self._count += 1
                self._prevSpeaking = time.time()

                self.callback({"type": "voiceActivity"})
            
            # Calculates the number of seconds that has passed since the last voice activity
            if self._prevSpeaking:
                self._speakingLength = time.time() - self._prevSpeaking

                # We make the speech length requirement higher when we're just starting
                if self._count == 0:
                    self.speechLength = self.startSpeechLength
                else:
                    self.speechLength = self.realSpeechLength
            
            if self._speakingLength > self.speechLength:
                self._isSpeech = False

            text, status, code = self.recognizer.recognize(bytes(data), self._isSpeech)

            if not self._isSpeech:
                self.callback({"type": "completeFrames", "frames": self._frames})
                self._reset()
            
            if code == "partial": self.callback({
                "type": "partialText",
                "text": text
            })
            elif code == "full": self.callback({
                "type": "fullText",
                "text": text
            })
            elif code == "unrecognized": self.callback({
                "type": "unrecognized"
            })
            elif code.startswith("error: "): self.callback({
                "type": "error",
                "text": code
            })
    
    def _start(self):
        with sd.RawInputStream(samplerate=RATE, channels=CHANNELS, blocksize=CHUNK, callback=self._recognize, dtype=np.int16) as _:
            while True:
                try:
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
    
    def start(self, block=True):

        """
        When called, the speech recognizer will start listening for audio
        Keep in mind if you didn't provide a callback this entire thing will be useless, 
            the recognized text and other stuff will be sent to the callback
        """

        if block:
            self._start()
        else:
            threading.Thread(target=self._start, daemon=True).start()

def main():

    def callback(data):
        print(data)

    vadModel = os.path.join(os.getcwd(), "models", "vad", "model.h5")

    vad = JarvisVAD(vadModel)
    recognizer = GoogleRecognizer("test.wav")
    speech = SpeechRecognizer(recognizer, vad, callback=callback)

    speech.start(block=True)

if __name__ == "__main__":
    main()
    