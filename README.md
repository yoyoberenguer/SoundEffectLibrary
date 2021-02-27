# SoundEffectLibrary

## DESCRIPTION
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
```    
## REQUIREMENT:
```

pip install pygame cython numpy librosa pyaudio matplotlib scipy wave pandas

- setuptools>=49.2.1
- pygame>=1.9.6
- Cython>=0.28
- numpy~=1.18.0
- matplotlib~=2.2.2
- scipy~=1.1.0
- Wave~=0.0.2
- PyAudio~=0.2.11
- pandas~=0.22.0
- librosa>=0.8.0 

- A compiler such visual studio, MSVC, CGYWIN setup correctly
  on your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install on your system 
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system, 
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/
```

## BUILDING PROJECT:
```
In a command prompt and under the directory containing the source files
C:\>python setup_project.py build_ext --inplace

If the compilation fail, refers to the requirement section and make sure cython 
and a C-compiler are correctly install on your system. 
```

### Capturing/Recording sound from a microphone
-----------------------------------------
```
Record sound(s) from the microphone to disk (default file is 'output.wav') when 
the variable record_ is set to True, otherwise return a buffer containing the data 
sample. If the variable duration_ is omitted, the record duration is cap to a maximum 
value of 10 seconds.
```
```python
# usage : 
record_microphone(duration_=10, record_=True)
data = record_microphone(duration_=10, record_=False)

```

```python

cpdef record_microphone(int format_=paInt16,
                        short int channels_=1,
                        int sample_rate_=44100,
                        int chunk_=16384,
                        int duration_=10,
                        bint record_=False,
                        str filename_="output.wav"):
"""
RECORD SOUNDS FROM THE MICROPHONE INTO A BUFFER 

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

### Record sound effect to disk (wav format)
```
Record a sound object to disk (pygame Sound object, such as pygame.sndarray.make_sound()).
The filename must be passed otherwise a ValueError will be raised (only WAV format are 
currently supported by version 1.0.1);, Return True when the file is successfully written to 
disk otherwise return False.
```
```python
# usage:
result_flag = record_sound(sound_object, 'output.wav')

```

```python
cpdef record_sound(sound_, str filename_):
    """
    SAVE A SOUND OBJECT ON DISK (INPUT CAN BE A NUMPY ARRAY OR A SOUND OBJECT)
    
    * input can be a pygame.mixer.sound object or a numpy array (monophonic or stereophonic)
    * compatible with stereophonic or monophonic sounds
    
    :param sound_: pygame.mixer.sound; Sound object to save onto disk 
    :param filename_: string; filename including file extension (this method is only compatible with wav file)
    :return: boolean; True | False if the sound has been successfully saved  
    """
 ```
### Data sample normalisation & Reverse normalisation 
```
It it sometimes required to normalised data sample before processing. 
This usually happen when the mixer is initialised in 16bit (signed or unsigned int mode) 
The below methods will convert monophonic and stereophonic 16bit audio sample into an 
32bit equivalent format. The normalisation is a straight forward calculation, as a result it can be 
used in real time processing without degrading performances of your game / application.
In addition you can use the reverse normalisation process to do the opposite when required
(32bit data sample converted into 16bit signed data sample).
```
```python
# usage for a monophonic audio sample signed 16bit format
normalize_array_mono(data)

# usage for a stereophonic audio sample signed 16bit format
normalize_array_stereo(data)

# usage for a sound object signed 16bit (monophonic or stereophonic modes) 
normalize_sound(sound)

# usage for inverse normalisation monophonic audio sample 32bit format
inverse_normalize_mono(data)

# usage for inverse normalisation stereophonic audio sample 32bit format
inverse_normalize_stereo(data)

# usage for inverse normalisation stereophonic audio sample 32bit format
inverse_normalize_stereo_asarray(data)

# usage for inverse normalisation monophonic audio sample 32bit format
inverse_normalize_mono_asarray(data)

```





