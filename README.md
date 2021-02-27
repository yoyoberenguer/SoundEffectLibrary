# SoundEffectLibrary

```
Sound Effect Library is a free software that include a large variety of tools to modify 
and create sound effects for video games and can also be used for sound processing. 
It provides fast algorithms written in python and Cython in addition to C/C++ 
code (external libraries) included with the project. 
This project rely on Pygame mixer and sndarray modules in order to build and extract
data samples into numpy.sndarray. The algorithms are build for int16 and float32 
array datatype and for monophonic & stereophonic sound effects
Choose the correct algorithm according to the data-type and sound model 

Below details list of methods available at your convenience
- Microphone recording 
- Sound recording (wav format)
- Data sample normalisation &reverse normalisation process 
- RMS calculator and display
- Fade in / Fade out effect
- Tinnitus effect
- Generate silence 
- Low pass filter 
- Harmonic representation
- Noise signal
- Square signal
- triangular signal 
- cosine signal 
- Cosine carrier 
- Sound time shifting
- Volume change 
- Reverse sound 
- Sound Inversion 
- Mixing sounds 
- Up / Down data sampling
- Panning sound effect 
- Median filtering 
- Averaging filtering 
- Echo sound effect
- Pitch shifting and time stretching  

 Not included in this version (1.0.1)
 - high pass / bandpass filtering
 - Gaussian filtering
    

Requirements : 
- Pygame 
- Numpy
- Librosa (for time stretching and pitch shifting)
- pyaudio (to record sound effect with microphone)
- Matplotlib (to represent graphically signal time domain to frequency domain, harmonics)
- scipy (for signal processing such as low pass/high pass filter and passband 
- wave (to create wav file to disk)
- Cython 

```


## Recording sound from a microphone
-----------------------------------------
```python
usage : 
record_microphone(duration_=10, record_=True)


```


```python

cpdef record_microphone(int format_=paInt16,
                        short int channels_=1,
                        int sample_rate_=44100,
                        int chunk_=16384,
                        int duration_=1,
                        bint record_=False,
                        str filename_="output.wav"):
"""
RECORD SOUNDS FROM THE MICROPHONE 

Return OSError if no microphone can be used for recording. 
OSError: [Errno -9996] Invalid input device (no default output device)

:param format_     : integer; Audio format paFloat32, paInt32, paInt24, paInt16, paInt8, paUInt8
:param channels_   : integer; number of channels must be 1 or 2    
:param sample_rate_: integer; sample rate must be in 8000, 11025, 16000, 22050, 32000, 44100, 48000,
                     88200, 96000, 176400, 192000, 352800, 384400
:param chunk_      : integer; Specifies the number of frames per buffer (default 16384)
:param duration_   : integer; Record duration, default 1 seconds
:param record_     : bool; True | False. If true the record will be save onto a file (default : output.wav)
:param filename_   : string; Record name (only when record is True) 
:return: Return a buffer type unsigned char shape (n,) 
"""
```
