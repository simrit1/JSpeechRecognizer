
# JSpeechRecognizer 
A speech recognizer library that combines VAD (Voice activity detection), Speech Recognition, and Wake word Recognition, the three must have functionality of a voice assistant.

The purpose of this library is to remove the hassle of setting up your own voice assistant ready speech recognizer.

## Installation
First install the pacakage with `pip install jspeechrecognizer`

Download the `model.h5` file inside the models folder (You're going to use this for the VAD)

## Usage Example
VAD model can be found inside the models folder

```py
from jspeechrecognizer import GoogleRecognizer, JarvisVAD, SpeechRecognizer

# Recognized speech and other stuff goes inside the callback function
def callback(data):

	"""
	It is important not to print the complete frames cause that's 
	going to take forever so we just filter it out
	"""
	
	if not data['type'] == "completeFrames":
		print(data)

# Create a VAD object
vadPath = "models/vad/model.h5" # can be found inside the models folder
vad = JarvisVAD(vadPath)

# Recognizer Object
recognizer = GoogleRecognizer("cache.wav") # GoogleRecognizer requires a temporary caching .wav file

# And finally the main SpeechRecognizer object
speech = SpeechRecognizer(recognizer, vad, callback=callback)

# Now we just start the engine and we're good to go
speech.start(block=False) # pass block=False if you don't want it to block

...
```
Say "jarvis, turn the lights off" to see it in action (obviously you can say anything but it must start with the wakeword)

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

## Recognizers
JSpeechRecognizer provides 2 recognizers out of the box.

### Vosk
First being the VoskRecognizer which uses [vosk-api](https://github.com/alphacep/vosk-api)
Vosk is capable of doing real time speech recognition without the need of an internet connection.
The only downside is all processing is done on your device (because of it being offline) requiring you to have a pretty beefy computer (depending on what model you use)

For a list of models go here https://alphacephei.com/vosk/models

#### Usage
```py
from jspeechrecognizer import VoskRecognizer
recognizer = VoskRecognizer(
	modelPath="path/to/vosk/model"
)

```
#### Parameters
`*modelPath`: path to your vosk model
`callback`: a callback function that takes one parameter

### Google
More specifically the `.recognize_google()` method from [speech_recognition](https://github.com/Uberi/speech_recognition)

A recognizer that uses google's speech recognition making it the most accurate and the least resource hungry, however it requires an internet connection

#### Usage
```py
from jspeechrecognizer import GoogleRecognizer
recognizer = GoogleRecognizer(
	cachePath="path/to/cache.wav"
)
```

#### Parameters
`*cachePath`: A .wav file to save the temporary .wav it creates
`callback`: A callback function that takes one parameter

## Voice Activity Detection
By default JSpeechRecognizer only offers one VAD class. But you can easily create your own by creating a class with a `.isSpeech` function  (check template below) that takes raw input stream `bytes` and returns either a `True` or `False`

### Template 
```py
class MyOwnVad:
	def __init__(self):
		pass # do anything you want here
	
	# The important function
	def isSpeech(self, stream: bytes):
		"""
		stream is usually the raw bytes you get from let's 
		say pyaudio's .read() function
		
		Do your processing here. All it has to return is
		a True or False boolean
		
		True if the stream is classified is being voice
		False if not
		"""
		return True
```

### JarvisVAD
Now let's talk about the default VAD of JSpeechRecognizer

This is not mean to be a replacement for webrtcvad but I've found that webrtcvad doesn't seem to filter out voice activity in the background (such as some music playing, television, etc)

This uses deep learning to predict whether a piece of audio is classified as speech or not.
It does this by checking the decibels of some frequencies in the provided stream and figuring out whether those decibels is considered as someone speaking or not.

NOTE: This may not be accurate depending on your situation, this has only been trained under my own environment. Feel free to use something like webrtcvad

#### Usage
```py
from jspeechrecognizer import JarvisVAD
vad = JarvisVAD(
	modelPath="path/to/model.h5"
)

```

#### Parameters
`*modelPath`: path to the model.h5 file
`callback`: A callback function that takes one parameter
`sensitivity`: How sensitive the prediction is, values below 0.90 don't really matter much but you can give it a try


## LICENSE
   Copyright 2021 Philippe Mathew

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

