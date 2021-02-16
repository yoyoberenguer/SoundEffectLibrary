# SoundEffectLibrary
Sound effect library 

```



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
