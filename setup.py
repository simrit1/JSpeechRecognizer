from distutils.core import setup
setup(
    name='jspeechrecognizer',
    packages=['jspeechrecognizer'], 
    version='0.1-beta', 
    license='apache-2.0',
    description='A Speech Recognition library that combines wakeword detection, speech recognition, and voice activity detection.',
    author='Philippe Mathew',
    author_email='philmattdev@gmail.com',
    url='https://github.com/bossauh/JSpeechRecognizer',
    download_url='https://github.com/bossauh/JSpeechRecognizer/archive/refs/tags/v_0.1-beta.tar.gz',
    keywords=['speech', 'recognition', 'vad', 'wakeword'],
    install_requires=[
        'pvporcupine',
        'vosk',
        'speechrecognition',
        'wave',
        'sounddevice',
        'tensorflow',
        'numpy'
    ],
    classifiers=[
        'Development Status :: 4 - Beta', # 3=Alpha | 4=Beta | 5=Production
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
