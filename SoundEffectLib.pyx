# cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True, optimize.unpack_method_calls=True, cdivision=True
# encoding: utf-8

from __future__ import print_function

__author__ = "Yoann Berenguer"
__copyright__ = "Copyright 2021."
__credits__ = ["Yoann Berenguer"]
__license__ = "MIT License"
__version__ = "1.0.1"
__maintainer__ = "Yoann Berenguer"
__email__ = "yoyoberenguer@hotmail.com"
__status__ = "tested"

"""
MIT License
Copyright (c) 2019 Yoann Berenguer
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

Sound Effect Library is a free software that include a large variety of tools to modify 
and create sound effects for video games and can also be used for sound processing. 
It provides fast algorithms written in python and Cython in addition to C/C++ code (external libraries) 
included with the project. 
This project rely on Pygame mixer and sndarray modules in order to build and extract data samples into numpy.sndarray. 
The algorithms are build for int16 and float32 array datatype and for monophonic & stereophonic sound effects
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

 Not included in this version 
 - Gaussian filtering
    
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

# In a command prompt and under the directory containing the source files
C:\>python setup_project.py build_ext --inplace
...
...
...
   Creating library build\temp.win-amd64-3.6\Release\SoundEffectLib.cp36-win_amd64.lib and object
    build\temp.win-amd64-3.6\Release\SoundEffectLib.cp36-win_amd64.exp
Generating code
Finished generating code


# If the compilation fail, refers to the requirement section and make sure cython 
# and a C-compiler are correctly install on your system. 

"""



"""
FLAG USED
@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

try:
    import pyaudio
except ImportError:
    raise ImportError("\n<pyaudio> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from pyaudio import paFloat32, paInt32, paInt24, paInt16, paInt8, paUInt8

try:
    import librosa
except ImportError:
    raise ImportError("\n<librosa> library is missing on your system."
          "\nTry: \n   C:\\pip install librosa on a window command prompt.")

try:
    import pygame
except ImportError:
    raise ImportError("\n<pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from pygame import sndarray

from pygame.sndarray import make_sound

from libc.stdio cimport printf
from libc.math cimport sqrt, cos, sin, log10, fabs, atan, atan2
from libc.stdlib cimport abs
from libc.limits cimport SHRT_MIN, SHRT_MAX
from libc.stdlib cimport malloc

try:
    cimport cython
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# from cpython.list cimport PyList_Append, PyList_GetItem, PyList_Size, PyList_SetItem, PyList_SET_ITEM
# from cpython.object cimport PyObject_SetAttr
# from cpython.dict cimport PyDict_SetItem
from cpython cimport PyObject_HasAttr, PyObject_IsInstance
# from cpython cimport array
from cython.parallel cimport prange

try:
    import numpy as numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

from numpy import zeros, int16, empty, asarray, float32, float64, float_, \
    average, amin, amax, round, fft, hanning, ascontiguousarray
cimport numpy
from numpy cimport uint8_t, int16_t, float32_t, complex_t, float64_t

try:
    from scipy import signal
except ImportError:
    raise ImportError("\n<scipy> library is missing on your system."
          "\nTry: \n   C:\\pip install scipy on a window command prompt.")

from ErrorMsg_uk import *

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("\n<matplotlib> library is missing on your system."
          "\nTry: \n   C:\\pip install matplotlib on a window command prompt.")

import logging

cdef extern from 'QuickSort.c' nogil:
    int * quickSort(int arr[], int low, int high)nogil
    float f_max(float arr[], int element)nogil

cdef extern from 'randnumber.c':
    void init_clock()nogil
    float randRangeFloat(float lower, float upper)nogil
    int randRange(int lower, int upper)nogil

cdef extern from 'PitchShifting.cpp':
    void smbPitchShift(float pitchShift, long numSampsToProcess, long fftFrameSize,
                       long osamp, float sampleRate, float *indata, float *outdata)nogil


DEF SCHEDULE = 'static'

DEF OPENMP = True
# num_threads – The num_threads argument indicates how many threads the team should consist of.
# If not given, OpenMP will decide how many threads to use.
# Typically this is the number of cores available on the machine. However,
# this may be controlled through the omp_set_num_threads() function,
# or through the OMP_NUM_THREADS environment variable.
if OPENMP is True:
    DEF THREAD_NUMBER = 8
else:
    DEF THREAD_NUMBER = 1


DEF PI = 3.14159265359
DEF PI2 = 2 * PI
DEF DEG_TO_RADIAN = PI / 180.0
DEF RADIAN_TO_DEG = 180.0 / PI
cdef:
    float INV_SHRT_MAX = 1.0 / SHRT_MAX
    float INV_SHRT_MIN = 1.0 / SHRT_MIN

# Sample rate allowed
FS = [8000, 11025, 16000, 22050, 32000, 44100, 48000,
      88200, 96000, 176400, 192000, 352800, 384400]

PYAUDIO_FORMAT = [paFloat32, paInt32, paInt24, paInt16, paInt8, paUInt8]

SOUNDTYPE = pygame.mixer.Sound

cdef struct rms:
    double s0
    double s1

ctypedef rms RMS

init_clock()


try:
    import wave
except ImportError:
    raise ImportError("\n<wave> library is missing on your system."
          "\nTry: \n   C:\\pip install wave on a window command prompt.")
import os

PATH = os.getcwd()

# todo profiling
cpdef record_microphone(int format_=paInt16,
                        short int channels_=1,
                        int sample_rate_=44100,
                        int chunk_=16384,
                        int duration_=10,
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

    if chunk_ == 0:
        raise ValueError("\nArgument chunk_ cannot be equal zero!")

    if chunk_<1024:
        raise ValueError(message35 % ("chunk_", 1024, chunk_) )

    if not format_ in PYAUDIO_FORMAT:
        raise ValueError("\nUnknown format %s, accept %s " % (format_, PYAUDIO_FORMAT))

    if channels_:
        if channels_ not in (1, 2):
            raise ValueError(message7)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if duration_ <= 0:
        raise ValueError(message35 % ("duration_", 1, duration_))

    name, extension = filename_.split(".")

    if len(extension) != 3:
        raise ValueError(message33)

    if extension.upper() != 'WAV':
        raise ValueError(message32 % extension)

    p = pyaudio.PyAudio()

    stream = p.open(format=format_,
                    channels=channels_,
                    rate=sample_rate_,
                    input=True,
                    frames_per_buffer=chunk_)

    print("* recording *")

    frames = []
    try:
        for _ in range(0, int(duration_ * sample_rate_ / chunk_)):
             frames.append(stream.read(chunk_))
    except Exception as e:
        logging.error(message36 % e)

    # Concatenate any number of bytes objects.
    frames_array = numpy.frombuffer(b''.join(frames), dtype=float32)

    print("* done recording *")

    stream.stop_stream()
    stream.close()

    if record_:
        try:
            wf = wave.open(filename_, 'wb')
            wf.setnchannels(channels_)
            wf.setsampwidth(p.get_sample_size(format_))
            wf.setframerate(sample_rate_)
            wf.writeframes(frames_array)
            wf.close()
        except Exception as e:
            logging.error(message37 % e)
            # Close the file
            if "wf" in globals():
                if wf is not None and hasattr(wf, "close"):
                    try:
                        wf.close()
                    except OSError as e:
                        logging.error(message38 % e)

            return

    p.terminate()
    return frames_array

# todo profiling
cpdef record_sound(sound_, str filename_):
    """
    SAVE A SOUND OBJECT ON DISK (INPUT CAN BE A NUMPY ARRAY OR A SOUND OBJECT)
    
    * input can be a pygame.mixer.sound object or a numpy array (monophonic or stereophonic)
    * compatible with stereophonic or monophonic sounds
    
    :param sound_: pygame.mixer.sound; Sound object to save onto disk 
    :param filename_: string; filename including file extension (this method is only compatible with wav file)
    :return: boolean; True | False if the sound has been successfully saved  
    """

    if not is_type_soundobject(sound_):
        # Convert array into a sound object
        try:
            sound_ = pygame.sndarray.make_sound(sound_)
        except:
            raise ValueError(message30)

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    if not os.path.exists(PATH):
        raise ValueError()

    name, extension = filename_.split(".")

    if len(extension) != 3:
        raise ValueError(message33)

    if extension.upper() != 'WAV':
        raise ValueError(message32 % extension)
    try:
        destination_file = wave.open(os.path.join(PATH, filename_), 'w')
        # set the parameters
        destination_file.setframerate(mixer_settings[0])  # frequency
        destination_file.setnchannels(mixer_settings[2])  # channel(s)
        destination_file.setsampwidth(2)
        # write raw PyGame sound buffer to wave file
        destination_file.writeframesraw(sound_.get_raw())
        destination_file.close()
        return True
    except Exception as e:
        logging.error(message37 % e)
        # Close the file
        if "destination_file" in globals():
            if destination_file is not None and hasattr(destination_file, "close"):
                try:
                    destination_file.close()
                except OSError as e:
                    logging.error(message38 % e)

        return False

cpdef normalize_array_mono(short [:] samples):
    """
    TAKE A NUMPY.NDARRAY AS INPUT (TYPE INT16) AND NORMALIZED THE VALUES (FLOAT32)
    
    :param samples: numpy.ndarray; Python numpy.ndarray type int16 representing the sound samples
    :return       : a memoryview of an numpy.ndarray, with values [-1.0 pass +1.0] type python float (cython.double). 
    Contiguous array
    """
    if not is_valid_mono_array(samples):
        raise ValueError(message11)

    cdef:
        int width = <object>samples.shape[0]
        float [::1] new_array = empty(width, float32)
        int i
        float s0

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples[i]
            if s0 > 0:
                new_array[i] = s0 * INV_SHRT_MAX
            elif s0 < 0:
                new_array[i] = -s0 * INV_SHRT_MIN
            else:
                new_array[i] = 0.0
    return new_array


cpdef float [:, :] normalize_array_stereo(short [:, :] samples_):
    """
    TAKE AN ARRAY INT16 AS INPUT (SOUND SAMPLES) AND RETURN A NORMALIZED SAMPLES (FLOAT32)
    
    :param samples_: ndarray; reference Sound samples into an array
    :return        : memoryview type [:, :] with floats values representing a normalized sound 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_.shape[0]
        int i
        float s0, s1
        float [:, :] new_array = empty((width, 2), float32)

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples_[i, 0]
            s1 = samples_[i, 1]
            if s0 > 0:
                new_array[i, 0] = s0 * INV_SHRT_MAX

            elif s0 < 0:
                new_array[i, 0] = -s0 * INV_SHRT_MIN

            else:
                new_array[i, 0] = 0.0

            if s1 > 0:
                new_array[i, 1] = s1 * INV_SHRT_MAX

            elif s1 < 0:
                new_array[i, 1] = -s1 * INV_SHRT_MIN

            else:
                new_array[i, 1] = 0.0

    return new_array


cpdef normalize_sound(sound_):
    """
    NORMALIZE A PYGAME SOUND OBJECT (STEREO OR MONOPHONIC), RETURN A NUMPY ARRAY 
    
    :param sound_: pygame.Sound; Pygame stereo sound object  
    :return      : Return a sndarray python array type (n, ) or (n, 2) object representing 
    a sound with float values [ -1.0 pass +1.0 ] 
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    # mono array
    if is_valid_mono_array(sound_array):
        # Array is already normalized
        if sound_array.dtype == float32:
            return sound_array

    # stereo array
    elif is_valid_stereo_array(sound_array):
        # Array is already normalized (float32)
        if sound_array.dtype==float32:
            return sound_array
    else:
        raise ValueError(message30)

    cdef:
        int channel_number = len(sound_array.shape)
        int width          = <object>sound_array.shape[0]
        short [::1] array_mono = sound_array if channel_number == 1 else empty(width, int16)
        short [:, :] array_stereo = sound_array if channel_number == 2 else empty((width, 2), int16)
        float [:, :] stereo_samples = empty((width, 2), float32)
        float [::1] mono_sample = empty(width, float32)
        int i
        float s0, s1

    if width == 0:
        raise ValueError(message12)

    # stereo
    if channel_number == 2:

        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                s0 = array_stereo[i, 0]
                s1 = array_stereo[i, 1]
                if s0 > 0:
                    stereo_samples[i, 0] = <float>(s0 * INV_SHRT_MAX)

                elif s0 < 0:
                    stereo_samples[i, 0] = <float>(-s0 * INV_SHRT_MIN)

                else:
                    stereo_samples[i, 0] = 0.0

                if s1 > 0:
                    stereo_samples[i, 1] = <float>(s1 * INV_SHRT_MAX)

                elif s1 < 0:
                    stereo_samples[i, 1] = <float>(-s1 * INV_SHRT_MIN)

                else:
                    stereo_samples[i, 1] = 0.0
        return asarray(stereo_samples)

    # mono
    elif channel_number == 1:

        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                s0 = array_mono[i]

                if s0 > 0:
                    mono_sample[i] = <float>(s0 * INV_SHRT_MAX)

                elif s0 < 0:
                    mono_sample[i] = <float>(-s0 * INV_SHRT_MIN)

                else:
                    mono_sample[i] = 0.0

        return asarray(mono_sample)

    else:
        raise ValueError(message30)



# **************************** FADE-IN *************************************

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in(sound_, float fade_in_, float sample_rate_):
    """
    FADE IN EFFECT (INPLACE)
    
    * Compatible monophonic and stereophonic sound effect
    
    :param sound_      : pygame Sound; Sound to fade-in (Monophonic or stereophonic)
    :param fade_in_    : float; end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_: float; Sample rate
    :return            : Void; change inplace
    """
    if not is_type_soundobject(sound_):
        raise ValueError(message23 % sound_)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39 % "sound_")

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    if sound_array.dtype == float32:
        if is_valid_mono_array(sound_array):
            sound_array = fade_in_mono_float32(sound_array, fade_in_, sample_rate_)

        elif is_valid_stereo_array(sound_array):
            sound_array = fade_in_stereo_float32(sound_array, fade_in_, sample_rate_)

        else:
            raise ValueError(message30)

    elif sound_array.dtype == int16:
        if is_valid_mono_array(sound_array):
            sound_array = fade_in_mono_int16(sound_array, fade_in_, sample_rate_)

        elif is_valid_stereo_array(sound_array):
            sound_array = fade_in_stereo_int16(sound_array, fade_in_, sample_rate_)

        else:
            raise ValueError(message30)
    else:
        raise ValueError(message27 % sound_array.dtype)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_mono_int16(short [::1] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR MONOPHONIC SOUND (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in).   
    the shape of the fade is linear, so it appears as a straight line from beginning
    to end. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with monophonic sound object only
    * Return a numpy.array shape (n, ) int16 with fade in effect 
    
    :param samples_     : ndarray; Reference Sound samples into an array. numpy.ndarray int16 
    (contiguous array) shape (n, )
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array contiguous shape (n, ) int16 with fade in effect 
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_, int16)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i] = <short>(samples_[i] * time_pos)

    return asarray(samples_)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_mono_float32(float [::1] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR MONOPHONIC SOUND (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in).   
    the shape of the fade is linear, so it appears as a straight line from beginning
    to end. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with monophonic sound object only
    * Return a numpy.array shape (n, ) float32 with fade in effect 
    
    :param samples_     : ndarray; Reference Sound samples into an array. numpy.ndarray float32 
    (contiguous array) shape (n, )
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array contiguous shape (n, ) float32 with fade in effect 
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_, float32)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i] = <float>(samples_[i] * time_pos)

    return asarray(samples_, float32)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_stereo_int16(short [:, :] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN STEREOPHONIC SOUNDS (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in).
    the shape of the fade is linear, so it appears as a straight line from beginning
    to end. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with stereophonic sound object only
    * Return a numpy.array shape (n, ) int16 with fade in effect 
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy ndarray shape (n, 2) int16 
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array stereophonic shape (n , 2) int16  
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i, 0] = <short>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <short>(samples_[i, 1] * time_pos)

    return asarray(samples_, int16)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_stereo_float32(float [:, :] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN STEREOPHONIC SOUNDS (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in).
    the shape of the fade is linear, so it appears as a straight line from beginning
    to end. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with stereophonic sound object only
    * Return a numpy.array shape (n, ) float32 with fade in effect 
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy ndarray shape (n, 2) float32 
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array stereophonic shape (n , 2) float32  
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_, float32)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i, 0] = <float>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <float>(samples_[i, 1] * time_pos)

    return asarray(samples_, float32)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_mono_inplace_int16(short [::1] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR MONOPHONIC SOUND (INT16) INPLACE
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with monophonic sound object only
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, ) monophonic int16
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Void
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i] = <short>(samples_[i] * time_pos)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_mono_inplace_float32(float [::1] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR MONOPHONIC SOUND (float32) INPLACE
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with monophonic sound object only
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, ) monophonic float32
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Void
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_, float32)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i] = <float>(samples_[i] * time_pos)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_stereo_inplace_int16(short [:, :] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR STEREOPHONIC SOUND (INT16) INPLACE
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with stereophonic sound object only
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) stereophonic int16
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Void
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i, 0], samples_[i, 1] = <short>(samples_[i, 0] * time_pos), <short>(samples_[i, 1] * time_pos)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_in_stereo_inplace_float32(float [:, :] samples_, const float fade_in_, const float sample_rate_):
    """
    FADE IN FOR STEREOPHONIC SOUND (float32) INPLACE
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
    
    * Compatible with stereophonic sound object only
    
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) stereophonic float32
    :param fade_in_     : float; fade_in_, end of the fade in effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_ : float; Sample rate
    :return             : Void
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_in_ == 0:
        return asarray(samples_, float32)
    elif fade_in_ < 0:
        raise ValueError(message24 % "fade_in_")
    elif fade_in_ > t:
        raise ValueError(message25 % ("fade_in_", t, fade_in_))

    cdef:
        int i = 0
        float time_pos
        int end = <int>(fade_in_ * sample_rate_)

    with nogil:
        for i in prange(0, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = 1 - <float>(end - i) / <float>end
            samples_[i, 0], samples_[i, 1] = <float>(samples_[i, 0] * time_pos), <float>(samples_[i, 1] * time_pos)

# ******************************************** FADE OUT ***********************************************************

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out(sound_, float fade_out_, float sample_rate_):
    """
    FADE OUT EFFECT (INPLACE)
    
    :param sound_      : pygame Sound; Sound to fade-out (monophonic or stereophonic)
    :param fade_out_   : float; start of the fade-out effect (in seconds). Cannot exceed the sound duration.
    :param sample_rate_: float; Sample rate
    :return            : Void
    """
    if not is_type_soundobject(sound_):
        raise ValueError(message23 % sound_)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39 % "sound_")

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")


    if sound_array.dtype == float32:
        if is_valid_mono_array(sound_array):
            fade_out_mono_float32(sound_array, fade_out_, sample_rate_)

        elif is_valid_stereo_array(sound_array):
            fade_out_stereo_float32(sound_array, fade_out_, sample_rate_)

        else:
            raise ValueError(message30)

    elif sound_array.dtype == int16:
        if is_valid_mono_array(sound_array):
            fade_out_mono_int16(sound_array, fade_out_, sample_rate_)

        elif is_valid_stereo_array(sound_array):
            fade_out_stereo_int16(sound_array, fade_out_, sample_rate_)

        else:
            raise ValueError(message30)
    else:
        raise ValueError(message27 % sound_array.dtype)



@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_mono_int16(short [::1] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT MONOPHONIC (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with monophonic sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array (numpy.ndarray with datatype int16,
                          buffer with un-normalized values
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array shape (n, ) int16 
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    if fade_out_ > t: raise ValueError(message25 % ("fade_out_", t, fade_out_))

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i] = <short>(samples_[i] * time_pos)

    return asarray(samples_, int16)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_mono_float32(float [::1] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT MONOPHONIC (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with monophonic sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array (numpy.ndarray with datatype float32,
                          buffer with un-normalized values
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : Return a numpy.array shape (n, ) float32 
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i] = <float>(samples_[i] * time_pos)

    return asarray(samples_, float32)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_stereo_int16(short [:, :] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT STEREOPHONIC (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with stereo sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) int16 
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : Return a Numpy.array shape (n, 2) int16 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i, 0] = <short>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <short>(samples_[i, 1] * time_pos)

    return asarray(samples_, int16)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_stereo_float32(float [:, :] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT STEREOPHONIC (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with stereo sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) float32 
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : Return a Numpy.array shape (n, 2) float32 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i, 0] = <float>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <float>(samples_[i, 1] * time_pos)

    return asarray(samples_, float32)

@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_mono_inplace_int16(short [::1] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT MONOPHONIC (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with monophonic sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.ndarray shape (n, ) int16 contiguous
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : void
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i] = <short>(samples_[i] * time_pos)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_mono_inplace_float32(float [::1] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT MONOPHONIC (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with monophonic sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.ndarray shape (n, ) float32 contiguous
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : void
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i] = <float>(samples_[i] * time_pos)


@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_stereo_inplace_float32(float [:, :] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT STEREOPHONIC (FLOAT32)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with stereo sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) float32 
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : void
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i, 0] = <float>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <float>(samples_[i, 1] * time_pos)



@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef fade_out_stereo_inplace_int16(short [:, :] samples_, const float fade_out_, const float sample_rate_):
    """
    FADE OUT STEREOPHONIC (INT16)
    
    These basic fades apply a fade to the selected audio such that the amplitude
    of the selection goes from absolute silence to the original amplitude (fade in)
    or from the original amplitude to absolute silence (fade out). 
    the shape of the fade is linear, so it appears as a straight line from beginning
    to ens. The speed of the fade in or out is therefore constant throughout its length
    and depends entirely on the length selected for the fade.
      
    * Compatible with stereo sound object only
     
    :param samples_     : ndarray; Reference Sound samples into an array. Numpy.array shape (n, 2) int16 
    :param fade_out_    : float; start time of the fade effect (in seconds). 
                          Fade out value cannot exceed the sound duration 
    :param sample_rate_ : float; Sample rate
    :return             : void
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float t = <float>width / <float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if fade_out_ >=t:
        return asarray(samples_)
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_in_")

    cdef:
        int i = 0
        float time_pos
        int start = <int>(fade_out_ * sample_rate_)

    with nogil:
        for i in prange(start, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            time_pos = <float>(width - i) / <float>(width - start)
            samples_[i, 0] = <short>(samples_[i, 0] * time_pos)
            samples_[i, 1] = <short>(samples_[i, 1] * time_pos)


cpdef tinnitus(float amplitude_ = 0.5,
               float duration_  = 0.1,
               float frequency_ = 6500,
               int sample_rate_ = 48000,
               float c_         = 0.0,
               double phi_      = 0.0
               ):
    """
    CREATE TINNITUS EFFECT (SOUND AFTER A LOUD EXPLOSION)
    
    :param amplitude_  : float; signal amplitude (normalized value [-1.0 pass 1.0] default 0.5 
    :param duration_   : float; duration in seconds default 1 second   
    :param frequency_  : float; frequency in hertz default 6500hz
    :param sample_rate_: integer; sample rate default 48khz
    :param c_          : float; Signal constant (default zero, centered on axis y) interval [-1.0 pass 1.0]
    :param phi_        : double; Radian offset (signal phase offset) interval ]-pi pass +pi[
    :return           : return tuple (sound vector and sound array)
    """

    if not (0.0 < amplitude_ <= 1.0):
        raise ValueError(message8)

    if duration_ <= 0.0:
        raise ValueError(message42 % "duration_")

    if frequency_ < 0.0:
        raise ValueError(message17)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not -1.0 <= c_ <= 1.0:
        raise ValueError(message18)

    if not (fabs(phi_) < 3.1416):
        raise ValueError(message19)

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        float omega = <float>(2.0 * numpy.pi * frequency_)
        float T = <float>(1.0 / sample_rate_)
        int length = <int>(duration_ * sample_rate_)
        int i = 0
        unsigned short int channel = mixer_settings[2]
        int mode = abs(mixer_settings[1])
        float [::1] y_mono = empty(length, float32)
        float [:, ::1] y_stereo = empty((length, 2), float32)
        short [::1] y_mono_int16 = empty(length, int16)
        short [:, ::1] y_stereo_int16 = empty((length, 2), int16)
        float y

    with nogil:
        if mode == 32:
            if channel == 1:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y_mono[i] = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))

            elif channel == 2:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    y_stereo[i, 0] = y
                    y_stereo[i, 1] = y

        elif mode == 16:
            if channel == 1:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    if y > 0:
                        y = y * <float>SHRT_MAX
                    elif y < 0:
                        y = -y * <float>SHRT_MIN
                    else:
                        y = 0.0
                    y_mono_int16[i] = <short>y

            elif channel == 2:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    if y > 0:
                        y = y * <float>SHRT_MAX
                    elif y < 0:
                        y = -y * <float>SHRT_MIN
                    else:
                        y = 0.0
                    y_stereo_int16[i, 0] = <short>y
                    y_stereo_int16[i, 1] = <short>y
        else:
            raise NotImplementedError(message7)

    if mode == 32:
        if channel == 1:
            return asarray(y_mono, dtype=float32)
        else:
            return asarray(y_stereo, dtype=float32)

    elif mode == 16:
        if channel == 1:
            return y_mono_int16
        elif channel == 2:
            return y_stereo_int16

@cython.optimize.unpack_method_calls(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.optimize.use_switch(False)
cpdef tinnitus_fade_out(float fade_out_,
                        float amplitude_ = 0.5,
                        float duration_  = 0.1,
                        float frequency_ = 6500,
                        int sample_rate_ = 48000,
                        float c_         = 0.0,
                        double phi_      = 0.0
                        ):
    """
    CREATE TINNITUS EFFECT WITH A FADE-OUT EFFECT (SOUND AFTER A LOUD EXPLOSION)
    :param fade_out_   : float; Fade out starting time effect (must be < to the sound effect length and >0)
    :param amplitude_  : float; signal amplitude (normalized value [-1.0 pass 1.0] default 0.5 
    :param duration_   : float; duration in seconds default 1 second   
    :param frequency_  : float; frequency in hertz default 6500hz
    :param sample_rate_: integer; sample rate default 48khz
    :param c_          : float; Signal constant (default zero, centered on axis y) interval [-1.0 pass 1.0]
    :param phi_        : double; Radian offset (signal phase offset) interval ]-pi pass +pi[
    :return           : return tuple (sound vector and sound array)
    """

    if not (0.0 < amplitude_ <= 1.0):
        raise ValueError(message8)

    if duration_ <= 0.0:
        raise ValueError(message42 % "duration_")

    if fade_out_ > duration_:
        raise ValueError(message43 % ("fade_out_", duration_))
    elif fade_out_ < 0:
        raise ValueError(message24 % "fade_out_")

    if frequency_ < 0.0:
        raise ValueError(message17)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not -1.0 <= c_ <= 1.0:
        raise ValueError(message18)

    if not (fabs(phi_) < 3.1416):
        raise ValueError(message19)

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        float omega = <float>(2.0 * numpy.pi * frequency_)
        float T = <float>(1.0 / sample_rate_)
        int length = <int>(duration_ * sample_rate_)
        int i = 0
        unsigned short int channel = mixer_settings[2]
        int mode = abs(mixer_settings[1])

        float [::1] y_mono = empty(length, float32)
        float [:, ::1] y_stereo = empty((length, 2), float32)

        short [::1] y_mono_int16 = empty(length, int16)
        short [:, ::1] y_stereo_int16 = empty((length, 2), int16)
        float y

    with nogil:
        if mode == 32:
            if channel == 1:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y_mono[i] = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))

            elif channel == 2:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    y_stereo[i, 0] = y
                    y_stereo[i, 1] = y

        elif mode == 16:
            if channel == 1:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    if y > 0:
                        y = y * <float>SHRT_MAX
                    elif y < 0:
                        y = -y * <float>SHRT_MIN
                    else:
                        y = 0.0
                    y_mono_int16[i] = <short>y

            elif channel == 2:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    if y > 0:
                        y = y * <float>SHRT_MAX
                    elif y < 0:
                        y = -y * <float>SHRT_MIN
                    else:
                        y = 0.0
                    y_stereo_int16[i, 0] = <short>y
                    y_stereo_int16[i, 1] = <short>y
        else:
            raise NotImplementedError(message7)

    if mode == 32:
        if channel == 1:
            return fade_out_mono_float32(y_mono, duration_ / 1000.0, sample_rate_)
        else:
            return fade_out_stereo_float32(y_stereo, duration_ / 1000.0, sample_rate_)

    if mode == 16:
        if channel == 1:
            return fade_out_mono_int16(y_mono_int16, duration_ / 1000.0, sample_rate_)
        elif channel == 2:
            return fade_out_stereo_int16(y_stereo_int16, duration_ / 1000.0, sample_rate_)


cpdef generate_silence(sound_, float start_, float end_, int sample_rate_):
    """
    MODIFY A MONOPHONIC | STEREOPHONIC SOUND BY INSERTING A SILENCE AT A SPECIFIC TIME INTERVAL (START, END)
     
    * The rest of the data samples remain unchanged. 
    
    :param sound_      : Sound; pygame sound object to modify (monophonic | stereophonic sound) 
    :param start_      : float; start time in seconds (sample number where the silence is starting)
    :param end_        : float; end time in seconds (sample number where the silence is ending
    :param sample_rate_: int; sample rate 
    :return            : Return a pygame sound with a silence inserted between start en end values, the sound model will 
    match the pygame mixer settings (mono or stereo)
    """
    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39 % 1)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array.shape[0]
        float t   = <float>width/<float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if not (0 <= end_ <= t):
        raise ValueError(message45 % (0, "end_", t, end_))

    if not (0 <= start_ < end_):
        raise ValueError(message44 % (0, "start_", end_, start_))

    if is_valid_mono_array(sound_array):
        pass
    elif is_valid_stereo_array(sound_array):
        pass
    else:
        raise ValueError(message30)


    cdef:
        int channel_number = len(sound_array.shape)

        short [::1] mono_int16 = sound_array if (channel_number==1 and sound_array.dtype == int16) \
            else empty(width, int16)
        float [::1] mono_float32 = sound_array if (channel_number==1 and sound_array.dtype == float32) \
            else empty(width, float32)

        short [:, :] stereo_int16 = sound_array if (channel_number==2 and sound_array.dtype == int16) \
            else empty((width, 2), int16)
        float [:, :] stereo_float32 = sound_array if (channel_number==2 and sound_array.dtype == float32) \
            else empty((width, 2), float32)

        int start = <int>(sample_rate_ * start_)                # sample's start
        int end   = <int>(sample_rate_ * end_)                  # sample's end
        int i

    if channel_number > 2:
        raise ValueError(message30)


    if sound_array.dtype == int16:

        if channel_number == 1:
            with nogil:
                for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    mono_int16[i] = 0
            return pygame.sndarray.make_sound(mono_int16)

        elif channel_number == 2:
            with nogil:
                for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    stereo_int16[i] = 0
            return pygame.sndarray.make_sound(stereo_int16)

    if sound_array.dtype == float32:
        if channel_number == 1:
            with nogil:
                for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    mono_float32[i] = 0.0
            return pygame.sndarray.make_sound(mono_float32)

        elif channel_number == 2:
            with nogil:
                for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    stereo_float32[i] = 0.0
            return pygame.sndarray.make_sound(stereo_float32)

    else:
        raise ValueError(message27 % sound_array.dtype)



cpdef generate_silence_array_mono_int16(short [:] sound_array_, float start_, float end_, int sample_rate_):
    """
    MODIFY A MONOPHONIC DATA SAMPLES ARRAY BY INSERTING A SILENCE AT A SPECIFIC TIME INTERVAL (INT16)
     
    The rest of the data samples remain unchanged. 
    
    :param sound_array_: numpy ndarray; Data samples (numpy ndarray datatype int16) representing the sound object
    :param start_: float; start time in seconds (sample number where the silence is starting)
    :param end_: float; end time in seconds (sample number where the silence is ending
    :param sample_rate_: int; sample rate 
    :return: Return the sound with a silence inserted between start and end values, the sound model will 
    match the pygame mixer settings (mono or stereo)
    """
    if not is_valid_mono_array(sound_array_):
        raise ValueError(message22 % 1)


    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array_.shape[0]
        float t   = <float>width/<float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if not (0 <= end_ <= t):
        raise ValueError(message45 % (0, "end_", t, end_))

    if not (0 <= start_ < end_):
        raise ValueError(message44 % (0, "start_", end_, start_))


    cdef:
        int start = <int>(sample_rate_ * start_)                # sample's start
        int end   = <int>(sample_rate_ * end_)                  # sample's end
        int i

    with nogil:
        for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i] = 0

    return pygame.sndarray.make_sound(sound_array_)



cpdef generate_silence_array_mono_float32(float [:] sound_array_, float start_, float end_, int sample_rate_):
    """
    MODIFY A MONOPHONIC DATA SAMPLES ARRAY BY INSERTING A SILENCE AT A SPECIFIC TIME INTERVAL (FLOAT32)
     
    The rest of the data samples remain unchanged. 
    
    :param sound_array_: numpy ndarray; Data samples (numpy ndarray datatype float32) representing the sound object
    :param start_: float; start time in seconds (sample number where the silence is starting)
    :param end_: float; end time in seconds (sample number where the silence is ending
    :param sample_rate_: int; sample rate 
    :return: Return the sound with a silence inserted between start and end values, the sound model will 
    match the pygame mixer settings (mono or stereo)
    """

    if not is_valid_mono_array(sound_array_):
        raise ValueError(message22 % 1)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array_.shape[0]
        float t   = <float>width/<float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if not (0 <= end_ <= t):
        raise ValueError(message45 % (0, "end_", t, end_))

    if not (0 <= start_ < end_):
        raise ValueError(message44 % (0, "start_", end_, start_))

    cdef:
        int start = <int>(sample_rate_ * start_)                # sample's start
        int end   = <int>(sample_rate_ * end_)                  # sample's end
        int i


    with nogil:
        for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i] = 0

    return pygame.sndarray.make_sound(sound_array_)


cpdef generate_silence_array_stereo_int16(short [:, :] sound_array_, float start_, float end_, int sample_rate_):
    """
    MODIFY A STEREOPHONIC DATA SAMPLES ARRAY BY INSERTING A SILENCE AT A SPECIFIC TIME INTERVAL (INT16)
     
    The rest of the data samples remain unchanged. 
    
    :param sound_array_: numpy ndarray; Data samples (numpy ndarray datatype int16) representing the sound object
    :param start_: float; start time in seconds (sample number where the silence is starting)
    :param end_: float; end time in seconds (sample number where the silence is ending
    :param sample_rate_: int; sample rate 
    :return: Return the sound with a silence inserted between start and end values, the sound model will 
    match the pygame mixer settings (mono or stereo)
    """
    if not is_valid_stereo_array(sound_array_):
        raise ValueError(message26 % 1)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array_.shape[0]
        float t   = <float>width/<float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if not (0 <= end_ <= t):
        raise ValueError(message45 % (0, "end_", t, end_))

    if not (0 <= start_ < end_):
        raise ValueError(message44 % (0, "start_", end_, start_))

    cdef:
        int start = <int>(sample_rate_ * start_)                # sample's start
        int end   = <int>(sample_rate_ * end_)                  # sample's end
        int i

    with nogil:
        for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i, 0] = 0
            sound_array_[i, 1] = 0

    return pygame.sndarray.make_sound(sound_array_)


cpdef generate_silence_array_stereo_float32(float [:, :] sound_array_, float start_, float end_, int sample_rate_):
    """
    MODIFY A STEREOPHONIC DATA SAMPLES ARRAY BY INSERTING A SILENCE AT A SPECIFIC TIME INTERVAL (FLOAT32)
     
    The rest of the data samples remain unchanged. 
    
    :param sound_array_: numpy ndarray; Data samples (numpy ndarray datatype float32) representing the sound object
    :param start_: float; start time in seconds (sample number where the silence is starting)
    :param end_: float; end time in seconds (sample number where the silence is ending
    :param sample_rate_: int; sample rate 
    :return: Return the sound with a silence inserted between start and end values, the sound model will 
    match the pygame mixer settings (mono or stereo)
    """
    if not is_valid_stereo_array(sound_array_):
        raise ValueError(message26 % 1)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>sound_array_.shape[0]
        float t   = <float>width/<float>sample_rate_

    if width == 0:
        raise ValueError(message12)

    if not (0 <= end_ <= t):
        raise ValueError(message45 % (0, "end_", t, end_))

    if not (0 <= start_ < end_):
        raise ValueError(message44 % (0, "start_", end_, start_))

    cdef:
        int start = <int>(sample_rate_ * start_)                # sample's start
        int end   = <int>(sample_rate_ * end_)                  # sample's end
        int i

    with nogil:
        for i in prange(start, end, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i, 0] = 0
            sound_array_[i, 1] = 0

    return pygame.sndarray.make_sound(sound_array_)



cpdef speed_up_array_stereo(short [:, :] samples_, int n_):
    raise NotImplementedError

cpdef gaussian():
    raise NotImplementedError

# todo profiling & testing
cpdef high_pass(sound_, float fc_):
    """
    APPLY A HIGH PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT 
    
    * This algorithm will return a sound object identical to the pygame mixer sound model
      (see pygame mixer initialization settings). 
      e.g : If you are passing a monophonic array as input and the pygame mixer is initialized in
      stereophonic then the algorithm will return a stereophonic sound object to match the 
      Pygame mixer settings, regardless of the the input data samples shape (n, ) or (n, 2).      
      When a extra channel is added e.g (conversion from a single track monophonic array into 
      a stereophonic model) the extra channel will be identical to the single channel with 5ms
      delay (same data type), see variable delay. 
      
    * The high pass filter will be apply to a single channel. 
          
    * when the Pygame mixer is initialized in int16, the data samples input will have to be 
      converted to float32 equivalent format for processing. This will slightly degrade the 
      performance of the algorithm compare to data sample model float32 bit. For best performance,
      initialized the Pygame mixer in float32 bit, monophonic mode. 
      
    :param sound_ : numpy.ndarray; Buffer shape (n, ) or (n, 2) of int16 or float32 
    :param fc_    : float; low cut frequency (hz)
    :return       : Return a sound object stereophonic or monophonic, depends on the pygame.mixer 
                    initialisation settings.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    # sound_array can be either monophonic or stereophonic depends
    # on the mixer setting initialization parameters
    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    # check array format (accept int16 and float32)
    # Normalized array int16 into float32 for processing
    if is_valid_mono_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_mono(sound_array)

    elif is_valid_stereo_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_stereo(sound_array)
    else:
        raise ValueError(message30)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'high')

    # filtfilt cannot take array shape (n, 2) as data samples
    # filtered_array is always a monophonic data sample (buffer array-like)
    if len(sound_array.shape) == 2:
        filtered_array = signal.filtfilt(b, a, sound_array[:, 0]).astype(dtype=float32)

    elif len(sound_array.shape) == 1:
        filtered_array = signal.filtfilt(b, a, sound_array).astype(dtype=float32)

    cdef:
        int width = sound_array.shape[0]
        float [::1] output_mono = empty(width, float32)
        float [:, :] output_stereo = zeros((width, 2), float32)
        float [::1] signal_ = filtered_array
        int i
        float delay = 0.005               # 5 ms delay between channel 1 and channel 2 (force delay for stereo mode)
        int s = <int>(frequency * delay)  # delay into the array for stereo mode only

    if width == 0:
        raise ValueError(message12)

    if len(sound_array.shape) == 2:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_stereo[i, 0] = signal_[i]
                # Introduce a delay
                if i + s <= width:
                    output_stereo[i + s, 1] = signal_[i]

        return pygame.sndarray.make_sound(output_stereo)

    elif len(sound_array.shape) == 1:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_mono[i] = signal_[i]
        return pygame.sndarray.make_sound(output_mono)

    else:
        raise ValueError(message30)


# todo profiling
cpdef band_pass(sound_, float fc_low, float fc_high, int order=5):
    """
    APPLY A BANDPASS FILTER WITH CUT FREQUENCY FC_LOW and FC_HIGH PASSED AS ARGUMENT 
    
    * This algorithm will return a sound object identical to the pygame mixer sound model
      (see pygame mixer initialization settings). 
      e.g : If you are passing a monophonic array as input and the pygame mixer is initialized in
      stereophonic then the algorithm will return a stereophonic sound object to match the 
      Pygame mixer settings, regardless of the the input data samples shape (n, ) or (n, 2).      
      When a extra channel is added e.g (conversion from a single track monophonic array into 
      a stereophonic model) the extra channel will be identical to the single channel with 5ms
      delay (same data type), see variable delay. 
      
    * The bandpass filter will be apply to a single channel. 
          
    * when the Pygame mixer is initialized in int16, the data samples input will have to be 
      converted to float32 equivalent format for processing. This will slightly degrade the 
      performance of the algorithm compare to data sample model float32 bit. For best performance,
      initialized the Pygame mixer in float32 bit, monophonic mode. 
    
    
    :param order  : Sound object; Order of the filter (default is 5). To achieve steeper slopes, 
                    higher-order filters are required.  
    :param fc_high: float; The high frequency cut - off  
    :param fc_low : float; The low pass frequency cut - off
    :param sound_ : numpy.ndarray; Buffer shape (n, ) or (n, 2) of int16, float32 representing the 
                    monophonic sound data
    :return       : Return a sound object stereophonic or monophonic, depends on the pygame.mixer 
                    initialisation settings.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if fc_low < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    # sound_array can be either monophonic or stereophonic depends
    # on the mixer setting initialization parameters
    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    # check array format (accept int16 and float32)
    # Normalized array int16 into float32 for processing
    if is_valid_mono_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_mono(sound_array)

    elif is_valid_stereo_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_stereo(sound_array)
    else:
        raise ValueError(message30)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int frequency = mixer_settings[0]
        float low =  2 * fc_low / frequency    # Normalize the frequency
        float high =  2 * fc_high / frequency

    if low < 0:
        low = 0
    elif low>1.0:
        low = 1.0
    if high < 0:
        high = 0
    elif high>1.0:
        high = 1.0

    b, a = signal.butter(order, [low, high], btype='band')

    # filtfilt cannot take array shape (n, 2) as data samples
    # filtered_array is always a monophonic data sample (buffer array-like)
    if len(sound_array.shape) == 2:
        filtered_array = signal.filtfilt(b, a, sound_array[:, 0]).astype(dtype=float32)

    elif len(sound_array.shape) == 1:
        filtered_array = signal.filtfilt(b, a, sound_array).astype(dtype=float32)

    cdef:
        int width = sound_array.shape[0]
        float [::1] output_mono = empty(width, float32)
        float [:, :] output_stereo = zeros((width, 2), float32)
        float [::1] signal_ = filtered_array
        int i
        float delay = 0.005               # 5 ms delay between channel 1 and channel 2 (force delay for stereo mode)
        int s = <int>(frequency * delay)  # delay into the array for stereo mode only

    if width == 0:
        raise ValueError(message12)

    if len(sound_array.shape) == 2:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_stereo[i, 0] = signal_[i]
                # Introduce a delay
                if i + s <= width:
                    output_stereo[i + s, 1] = signal_[i]

        return pygame.sndarray.make_sound(output_stereo)

    elif len(sound_array.shape) == 1:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_mono[i] = signal_[i]
        return pygame.sndarray.make_sound(output_mono)

    else:
        raise ValueError(message30)


cpdef low_pass_mono_inplace_int16(short [::1] sound_array_, float fc_):
    """
    APPLY A LOW PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT 
    
    :param sound_array_: numpy.ndarray; Buffer shape (n, ) of int16 representing the monophonic sound data
    :param fc_         : float; low cut frequency (hz) cannot be < 0.0
    :return            : Void; Changes apply inplace
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    if not is_valid_mono_array(sound_array_):
        raise ValueError(message27 % sound_array_.dtype)

    cdef:
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'low')

    cdef:
        # normalized sound_array_ values before call and convert to float32
        float [:] low_pass_array = signal.filtfilt(b, a, normalize_array_mono(sound_array_)).astype(dtype=float32)
        int length = sound_array_.shape[0]
        int i
        float s0

    with nogil:
        for i in prange(length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = <float>low_pass_array[i]
            if s0 > 0:
                s0 = s0 * <float>SHRT_MAX
            elif s0 < 0:
                s0 = -s0 * <float>SHRT_MIN
            else:
                s0 = 0.0
            sound_array_[i] = <short>s0


cpdef low_pass_mono_inplace_float32(float [::1] sound_array_, float fc_):
    """
    APPLY A LOW PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT
    
    :param sound_array_: numpy.ndarray; Buffer shape (n, ) of float32 representing the monophonic sound data
    :param fc_         : float; low cut frequency (hz) cannot be < 0.0
    :return            : Void; Changes apply inplace
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    if not is_valid_mono_array(sound_array_):
        raise ValueError(message27 % sound_array_.dtype)

    cdef:
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'low')

    cdef:
        # normalized sound_array_ values before call and convert to float32
        double [:] low_pass_array = signal.filtfilt(b, a, sound_array_)
        int length = sound_array_.shape[0]
        int i

    with nogil:
        for i in prange(length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i] = <float>low_pass_array[i]


cpdef low_pass_stereo_inplace_int16(short [:, ::1] sound_array_, float fc_):
    """
    APPLY A LOW PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT
    
    :param sound_array_: numpy.array; array shape (n, 2) int16 representing the stereophonic data samples
    :param fc_         : float; low cut frequency (hz) cannot be < 0.0
    :return            : Void; Changes apply inplace
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    if not is_valid_stereo_array(sound_array_):
        raise ValueError(message27 % sound_array_.dtype)

    cdef:
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'low')

    cdef:
        int length = sound_array_.shape[0]
        float [:] channel0 = signal.filtfilt(b, a, normalize_array_mono(sound_array_[:, 0])).astype(dtype=float32)

        int i
        float s0, s1

    with nogil:
        for i in prange(length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = <float>channel0[i]
            if s0 > 0:
                s0 = s0 * <float>SHRT_MAX
            elif s0 < 0:
                s0 = -s0 * <float>SHRT_MIN
            else:
                s0 = 0.0
            sound_array_[i, 0] = <short>s0
            sound_array_[i, 1] = <short>s0


cpdef low_pass_stereo_inplace_float32(float [:, ::1] sound_array_, float fc_):
    """
    APPLY A LOW PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT
    
    :param sound_array_: numpy.array; array shape (n, 2) float32 representing the stereophonic data samples
    :param fc_         : float; low cut frequency (hz) cannot be < 0.0
    :return            : Void; Changes apply inplace
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    if not is_valid_stereo_array(sound_array_):
        raise ValueError(message27 % sound_array_.dtype)

    cdef:
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'low')

    cdef:
        int length = sound_array_.shape[0]
        double [:] channel0 = signal.filtfilt(b, a, sound_array_[:, 0])
        int i

    with nogil:
        for i in prange(length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            sound_array_[i, 0] = <float>channel0[i]
            sound_array_[i, 1] = <float>channel0[i]


cpdef low_pass(sound_, float fc_):
    """
    APPLY A LOW PASS FILTER WITH CUT FREQUENCY FC PASSED AS ARGUMENT
    
    * This algorithm will return a sound object identical to the pygame mixer sound model
      (see pygame mixer initialization settings). 
      e.g : If you are passing a monophonic array as input and the pygame mixer is initialized in
      stereophonic then the algorithm will return a stereophonic sound object to match the 
      Pygame mixer settings, regardless of the the input data samples shape (n, ) or (n, 2).      
      When a extra channel is added e.g (conversion from a single track monophonic array into 
      a stereophonic model) the extra channel will be identical to the single channel with 5ms
      delay (same data type), see variable delay. 
      
    * The low pass filter will be apply to a single channel. 
          
    * when the Pygame mixer is initialized in int16, the data samples input will have to be 
      converted to float32 equivalent format for processing. This will slightly degrade the 
      performance of the algorithm compare to data sample model float32 bit. For best performance,
      initialized the Pygame mixer in float32 bit, monophonic mode. 
    
    :param sound_: Pygame sound; Pygame Sound object (stereophonic or monophonic sound)
    :param fc_   : float low pass cut off frequency
    :return      : sound object; Return a pygame Sound with a low pass filter effect apply to the data samples, 
                   monophonic or stereophonic, depends on the pygame mixer initialisation settings
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if fc_ < 0:
        raise ValueError(message24 % "fc frequency cut")

    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    # sound_array can be either monophonic or stereophonic depends
    # on the mixer setting initialization parameters
    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    # check array format (accept int16 and float32)
    # Normalized array int16 into float32 for processing
    if is_valid_mono_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_mono(sound_array)

    elif is_valid_stereo_array(sound_array):
        if sound_array.dtype == int16:
            sound_array = normalize_array_stereo(sound_array)
    else:
        raise ValueError(message30)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int frequency = mixer_settings[0]
        float w =  2 * fc_ / frequency    # Normalize the frequency

    if w < 0:
        w = 0
    elif w>1.0:
        w = 1.0

    b, a = signal.butter(5, w, 'low')

    # filtfilt cannot take array shape (n, 2) as data samples
    # filtered_array is always a monophonic data sample (buffer array-like)
    if len(sound_array.shape) == 2:
        filtered_array = signal.filtfilt(b, a, sound_array[:, 0]).astype(dtype=float32)

    elif len(sound_array.shape) == 1:
        filtered_array = signal.filtfilt(b, a, sound_array).astype(dtype=float32)

    cdef:
        int width = sound_array.shape[0]
        float [::1] output_mono = empty(width, float32)
        float [:, :] output_stereo = zeros((width, 2), float32)
        float [::1] signal_ = filtered_array
        int i
        float delay = 0.005               # 5 ms delay between channel 1 and channel 2 (force delay for stereo mode)
        int s = <int>(frequency * delay)  # delay into the array for stereo mode only

    if width == 0:
        raise ValueError(message12)

    if len(sound_array.shape) == 2:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_stereo[i, 0] = signal_[i]
                # Introduce a delay
                if i + s <= width:
                    output_stereo[i + s, 1] = signal_[i]

        return pygame.sndarray.make_sound(output_stereo)

    elif len(sound_array.shape) == 1:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                output_mono[i] = signal_[i]
        return pygame.sndarray.make_sound(output_mono)

    else:
        raise ValueError(message30)


cpdef harmonics(object samples_:numpy.ndarray, int sampling_rate_=44100, int width=255, int height=255):
    """
    SIGNAL PERIODICITY ANALYSIS (FOURIER SPECTRUM). 
    
    This method return a pygame surface representing the signal frequency spectrum, by default the 
    surface is set to 255x255 pixels without alpha transparency layer (24 bits format)
    
    * Argument 1 compatible with numpy.array type float_, float32 and float64 shape (n, 2), compatible 
      stereophonic or monophonic sound data samples
    
    * Spectrum analysis will be performed on channel 0 only 
    
    * To convert the image to fast blit with pygame convert() method, the pygame display 
      must be initialized prior calling this function, otherwise the method will return 
      a pygame surface un-optimize for blit
      
    * This method is relatively slow (26ms)
    
    :param samples_      : numpy.ndarray; Data samples representing a sound (stereo | mono) effect (numpy ndarray 
    type float_, float32 or float64). Int16 datatype will be converted to float equivalent before processing 
    :param sampling_rate_: integer; default 44100 hz 
    :param width         : int; width of the surface 
    :param height        : int; height of the surface 
    :return              : pygame.Surface; Surface (graph) representing the frequency 
    analysis of the samples (amplitude vs frequencies) and array representing data
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # compatible float and double
    if is_monophonic(samples_):
        pass
    elif is_stereophonic(samples_):
        pass
    else:
        raise ValueError(message30)

    if not sampling_rate_ in FS:
        raise ValueError(message10 % (sampling_rate_, FS))

    cdef int length=<object>samples_.shape[0]

    if length ==0:
        raise ValueError(message2)

    cdef int channel_number =  len(samples_.shape)

    # convert double to float
    if samples_.dtype == numpy.float64:

        if channel_number == 1:
            samples_ = numpy.asarray(samples_, dtype=float32)

        elif channel_number== 2:
            # we need channel 0 for analysis
            samples_ = numpy.asarray(samples_[:, 0], dtype=float32)

    # Normalize array (stereo or mono) type int16 into equivalent float32
    elif samples_.dtype == numpy.int16:

        if channel_number == 1:
            samples_ = normalize_array_mono(samples_)

        elif channel_number == 2:
            # we need channel 0 for analysis
            samples_ = normalize_array_stereo(samples_)[0]

        else:
            raise ValueError(message30)

    # float32
    else:
        if channel_number == 2:
            samples_ = samples_[:, 0]  # Extract channel zero only

    cdef:
        float period = 1.0 / <float>sampling_rate_
        t_vec = numpy.arange(length) * period
        numpy.ndarray[complex_t, ndim=1] Y_k = fft.fft(samples_)[0:<int>(<float>length / 2.0)] / <float>length

    Y_k[1:] = 2 * Y_k[1:]
    cdef numpy.ndarray[float32_t, ndim=1] pxx = numpy.abs(Y_k).astype(dtype=float32)
    cdef numpy.ndarray[float32_t, ndim=1] f = \
        <float>sampling_rate_ * numpy.arange((<float>length / 2.0), dtype=float32) / <float>length

    # plotting
    fig, ax = plt.subplots()
    plt.plot(f, pxx, linewidth=0.9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [Hz]')

    fig.canvas.draw()
    data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(1, 0, 2)
    surf = pygame.surfarray.make_surface(data)
    # Smoothscale is compatible with 24 and 32bit image only
    surf = pygame.transform.smoothscale(surf, (width, height)).convert() \
        if pygame.display.get_init() else pygame.transform.smoothscale(surf, (width, height))
    return surf, pxx


# ******************************************** REMOVE SILENCE(S) ***********************************************

cpdef remove_silence_stereo_int16(short [:, :] samples_, rms_threshold_=None, bint bypass_avg_=False):
    """
    TRIM LEADING AND TRAILING SILENCE FROM AN AUDIO SIGNAL (STEREOPHONIC INT16)
    
    * Compatible stereophonic sample data only 
    
    :param samples_      : numpy.array; Represent the data samples (stereophonic) type int16 shape (n, 2)
    :param rms_threshold_: float; Threshold RMS (in decibels), below this value, samples are considered unable
     to be heard and will be removed from the final data. Value must be < signal average db
    :param bypass_avg_   : bool; Bypass the average value (RMS value > avg can be considered negligible)
    :return              : Return a pygame.mixer.Sound object (stereophonic) 
    """

    if rms_threshold_ is None:
        raise ValueError(message13)

    cdef float threshold = rms_threshold_

    if not is_valid_stereo_array(samples_):
        raise ValueError(message4 % str(samples_.shape))

    cdef:
        int width = <object>samples_.shape[0]
        int count = 0
        int i = 0, j = width, k = 0
        float [:, :] normalized_samples = normalize_array_stereo(samples_)
        float rms_value_0, rms_value_1
        float max_db0, max_db1, avg_left_db, avg_right_db, avg_centre_db
        bint silence_start = False, silence_end = False

    if width == 0:
        raise ValueError(message12)

    avg_left_db, avg_right_db, avg_centre_db = rms_values_stereo(samples_)

    if not bypass_avg_:
        if threshold > avg_centre_db:
            raise ValueError(message3 % (threshold, avg_centre_db))

    with nogil:
        # Skip the start if first entry RMS value > threshold (no silence)
        if <float>(10.0 * log10(normalized_samples[0, 0] ** 2)) < threshold and \
            <float>(10.0 * log10(normalized_samples[0, 1] ** 2)) < threshold:

            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value_0 = <float>(10.0 * log10(normalized_samples[i, 0] ** 2))
                rms_value_1 = <float>(10.0 * log10(normalized_samples[i, 1] ** 2))
                if rms_value_0 > threshold and rms_value_1 > threshold:
                    silence_start = True
                    break

        if <float>(10.0 * log10(normalized_samples[width - 1, 0] ** 2)) < threshold and \
            <float>(10.0 * log10(normalized_samples[width - 1, 1] ** 2)) < threshold:

            for j in prange(width-1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

                rms_value_0 = <float>(10.0 * log10(normalized_samples[j, 0] ** 2))
                rms_value_1 = <float>(10.0 * log10(normalized_samples[j, 1] ** 2))
                # printf("\nALL  J %f %f %f %d %i",\
                #     normalized_samples[j, 0], rms_value_0, threshold, rms_value_0 > threshold, j)

                if rms_value_0 > threshold and rms_value_1 > threshold:
                    silence_end = True
                    # change j (current value is > threshold and will be kept).
                    # Previous value is j + 1 as we are decrementing j.
                    j = j + 1
                    break

    # No silence, return original data samples
    if not (silence_start | silence_end): return make_sound(numpy.asarray(samples_, int16))

    if j - i <=0:
        return make_sound(numpy.zeros((width, 2), int16))

    cdef float [:, :] new_array = zeros((j - i, 2), float32)

    with nogil:
        for k in prange(i, j):
            new_array[k - i, 0] = normalized_samples[k, 0]
            new_array[k - i, 1] = normalized_samples[k, 1]

    return inverse_normalize_stereo(numpy.asarray(new_array))


cpdef remove_silence_stereo_float32(float [:, :] samples_, rms_threshold_=None, bint bypass_avg_=False):
    """
    TRIM LEADING AND TRAILING SILENCE FROM AN AUDIO SIGNAL (STEREOPHONIC float32)
    
    * Compatible stereophonic sample data only (float32)
    
    :param samples_      : numpy.array; Represent the data samples (stereophonic) type float32 shape (n, 2)
    :param rms_threshold_: float; Threshold RMS (in decibels), below this value, samples are considered unable
     to be heard and will be removed from the final data. Value must be < signal average db
     :param bypass_avg_   : bool; Bypass the average value (RMS value > avg can be considered negligible)
    :return              : Return a pygame.mixer.Sound object (stereophonic) 
    """

    if rms_threshold_ is None:
        raise ValueError(message13)

    cdef float threshold = rms_threshold_

    if not is_valid_stereo_array(samples_):
        raise ValueError(message4 % str(samples_.shape))

    cdef:
        int width = <object>samples_.shape[0]
        int count = 0
        int i = 0, j = width, k = 0
        float rms_value_0, rms_value_1
        float max_db0, max_db1, avg_left_db, avg_right_db, avg_centre_db
        bint silence_start = False, silence_end = False

    if width == 0:
        raise ValueError(message12)

    # # NOT USED
    # max_db0, max_db1 = rms_max_stereo(normalized_samples)
    avg_left_db, avg_right_db, avg_centre_db = rms_values_stereo(samples_)

    # RMS Threshold > avg_db is considered an invalid value
    if not bypass_avg_:
        if threshold > avg_centre_db:
            raise ValueError(message3 % (threshold, avg_centre_db))

    with nogil:
        # Skip the start if first entry RMS value > threshold (no silence)
        if <float>(10.0 * log10(samples_[0, 0] ** 2)) < threshold and \
            <float>(10.0 * log10(samples_[0, 1] ** 2)) < threshold:

            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value_0 = <float>(10.0 * log10(samples_[i, 0] ** 2))
                rms_value_1 = <float>(10.0 * log10(samples_[i, 1] ** 2))
                if rms_value_0 > threshold and rms_value_1 > threshold:
                    silence_start = True
                    break

        if <float>(10.0 * log10(samples_[width - 1, 0] ** 2)) < threshold and \
            <float>(10.0 * log10(samples_[width - 1, 1] ** 2)) < threshold:

            for j in prange(width-1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

                rms_value_0 = <float>(10.0 * log10(samples_[j, 0] ** 2))
                rms_value_1 = <float>(10.0 * log10(samples_[j, 1] ** 2))
                # printf("\nALL  J %f %f %f %d %i",\
                #     normalized_samples[j, 0], rms_value_0, threshold, rms_value_0 > threshold, j)

                if rms_value_0 > threshold and rms_value_1 > threshold:
                    silence_end = True
                    # change j (current value is > threshold and will be kept).
                    # Previous value is j + 1 as we are decrementing j.
                    j = j + 1
                    break

    # No silence, return original data samples
    if not (silence_start | silence_end): return make_sound(numpy.asarray(samples_, int16))

    if j - i <=0:
        return make_sound(numpy.zeros((width, 2), int16))

    cdef float [:, :] new_array = zeros((j - i, 2), float32)

    with nogil:
        for k in prange(i, j):
            new_array[k - i, 0] = samples_[k, 0]
            new_array[k - i, 1] = samples_[k, 1]

    return make_sound(numpy.asarray(new_array))

cpdef remove_silence_mono_int16(short [::1] samples_, rms_threshold_=None, bint bypass_avg_ = False):
    """
    TRIM LEADING AND TRAILING SILENCE FROM AN AUDIO SIGNAL (MONOPHONIC INT16)
    
    * Compatible monophonic sample data only 
    
    :param samples_      : numpy.array; Represent the data samples (monophonic) type int16 buffer
    :param rms_threshold_: float; Threshold RMS (in decibels), below this value samples are considered unable
     to be heard and will be removed from the final sampling. Value must be < signal average db
    :param bypass_avg_   : bool; Bypass the average value (RMS value > avg can be considered negligible)
    :return              : Return a pygame.mixer.Sound object (monophonic)
    """
    if rms_threshold_ is None:
        raise ValueError(message13)

    cdef float threshold = rms_threshold_

    if not is_valid_mono_array(samples_):
        raise ValueError(message0 % str(samples_.shape))

    cdef:
        int width = <object>samples_.shape[0]
        float [::1] normalized_array = normalize_array_mono(samples_)
        int count = 0
        int i = 0, j = width, k =0
        float rms_value
        float avg_db
        bint silence_start, silence_end

    if width == 0:
        raise ValueError(message12)

    avg_db = rms_value_mono(samples_)

    if not bypass_avg_:
        if threshold > avg_db:
            raise ValueError(message3 % (threshold, avg_db))

    with nogil:

        if <float>(10.0 * log10(normalized_array[0] ** 2)) < threshold:

            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value = <float>(10.0 * log10(normalized_array[i] ** 2))
                if rms_value >= threshold:
                    silence_start = True
                    break

        if <float>(10.0 * log10(normalized_array[width - 1] ** 2)) < threshold:

            for j in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value = <float>(10.0 * log10(normalized_array[j] ** 2))
                if rms_value >= threshold:
                    silence_end = True
                    break

    # No silence, return original data samples
    if not (silence_start | silence_end):
        return make_sound(numpy.asarray(samples_, int16))

    if j - i <=0:
        return make_sound(numpy.zeros(width, int16))

    cdef float [::1] new_array = empty(j - i, float32)

    with nogil:
        for k in prange(i, j):
            new_array[k - i] = normalized_array[k]

    return inverse_normalize_mono(numpy.asarray(new_array, dtype=float32))


cpdef remove_silence_mono_float32(float [::1] samples_, rms_threshold_=None, bint bypass_avg_ = False):
    """
    TRIM LEADING AND TRAILING SILENCE FROM AN AUDIO SIGNAL (MONOPHONIC FLOAT32)
    
    * Compatible monophonic sample data only float32
    
    :param samples_      : numpy.array; Represent the data samples (monophonic) type float32 buffer
    :param rms_threshold_: float; Threshold RMS (in decibels), below this value samples are considered unable
     to be heard and will be removed from the final sampling. Value must be < signal average db
    :param bypass_avg_   : bool; Bypass the average value (RMS value > avg can be considered negligible)
    :return              : Return a pygame.mixer.Sound object (monophonic)
    """
    if rms_threshold_ is None:
        raise ValueError(message13)

    cdef float threshold = rms_threshold_

    if not is_valid_mono_array(samples_):
        raise ValueError(message0 % str(samples_.shape))

    cdef:
        int width = <object>samples_.shape[0]
        int count = 0
        int i = 0, j = width, k =0
        float rms_value
        float avg_db
        bint silence_start, silence_end

    if width == 0:
        raise ValueError(message12)

    avg_db = rms_value_mono(samples_)

    if not bypass_avg_:
        if threshold > avg_db:
            raise ValueError(message3 % (threshold, avg_db))

    with nogil:

        if <float>(10.0 * log10(samples_[0] ** 2)) < threshold:

            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value = <float>(10.0 * log10(samples_[i] ** 2))
                if rms_value >= threshold:
                    silence_start = True
                    break

        if <float>(10.0 * log10(samples_[width - 1] ** 2)) < threshold:

            for j in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                rms_value = <float>(10.0 * log10(samples_[j] ** 2))
                if rms_value >= threshold:
                    silence_end = True
                    break

    # No silence, return original data samples
    if not (silence_start | silence_end):
        return make_sound(numpy.asarray(samples_))

    if j - i <=0:
        return make_sound(numpy.zeros(width, int16))

    cdef float [::1] new_array = empty(j - i, float32)

    with nogil:
        for k in prange(i, j):
            new_array[k - i] = samples_[k]

    return make_sound(numpy.asarray(new_array))


# ********************************************* SIGNALS ********************************************************

cpdef noise_signal(
        float amplitude_ = 1.0,
        float duration_  = 0.5,
        int sample_rate_ = 44100,
        ):
    """
    CREATE A SOUND GENERATING NOISE 
    
    In experimental sciences, noise can refer to any random fluctuations of data that hinders perception of a signal.
    
    * Pygame mixer has to be initialized prior calling this method.
    * Noise sound will be either monophonic or stereophonic (depends on Pygame mixer settings) 
    
    :param amplitude_  : float; Signal amplitude peak to peak, default 1.0 (must be in range ]0.0 pass 1.0])
    :param duration_   : float; duration, time in seconds
    :param sample_rate_: integer; sample rate
    :return: pygame.mixer.Sound object;  Return a noise sound (random values generated). The sound object created 
    with make_sound method will be either stereo or monophonic. Check the pygame.mixer settings channel before calling
    this function.
    """

    if not (0.0 < fabs(amplitude_) <= 1.0) :
        raise ValueError(message8)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not duration_ > 0:
        raise ValueError(message16)

    # (frequency, format, channels)
    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int format_ = mixer_settings[1]
        int width = <int>(duration_ * sample_rate_)
        float [::1] noise_array_nono = empty(width, float32)
        float [:, ::1] noise_array_stereo = empty((width, 2), float32)
        int i = 0
        float r = 0.0

    # random generator
    init_clock()

    with nogil:
        if channel == 1:
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                noise_array_nono[i] = randRangeFloat(-amplitude_, amplitude_)

        elif channel == 2:
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                r = randRangeFloat(-amplitude_, amplitude_)
                noise_array_stereo[i, 0] = r
                noise_array_stereo[i, 1] = r
        else:
            raise NotImplementedError(message7)

    if channel == 1:
        if format_  == -16:
            return make_sound(numpy.asarray(inverse_normalize_mono(noise_array_nono)))
        else:
            return make_sound(numpy.asarray(noise_array_nono))

    else:
        if format_ == -16:
            return make_sound(numpy.asarray(inverse_normalize_stereo(noise_array_stereo)))
        else:
            return make_sound(numpy.asarray(noise_array_stereo))


cpdef square_signal(float amplitude_ = 1.0,
                    float duration_  = 1.0,
                    float frequency_ = 100,
                    int sample_rate_ = 48000,
                    float c_         = 0.0,
                    float phi_       = 0.0
                    ):
    """
    RETURN A SQUARE DATA SAMPLE 
    
    * Amplitude zero will return a numpy array filled with zeros
    
    * Frequency = 0.0 will return a continuous signal (-1.0/+1.0 (float32) | 32767/-32768 (int16) + offset 
    if any, signed by the amplitude value)  
    
    :param amplitude_  : float; signal amplitude (normalized value [-1.0 pass 1.0]
    :param duration_   : float; duration in seconds default 1 second   
    :param frequency_  : float; frequency in hertz default 1khz
    :param sample_rate_: integer; sample rate default 48khz
    :param c_          : float; Signal constant (default zero, centered on axis y)
    :param phi_        : float; Radian offset (signal phase offset)
    :return           : return tuple (sound vector and sound array)
    """
    assert fabs(amplitude_) <= 1.0, "\nArgument amplitude should be in range [-1.0pass 1.0] "

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if duration_ <= 0.0:
        raise ValueError(message16)

    if frequency_<0:
        raise ValueError(message17)

    if not -1.0 <= c_ <= 1.0:
        raise ValueError(message18)

    if not fabs(phi_) <= PI:
        raise ValueError(message19)

    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int format_ = mixer_settings[1]
        float omega = 2 * numpy.pi * frequency_
        float T = 1.0 / sample_rate_
        numpy.ndarray[float32_t, ndim=1] t_vec = (numpy.arange(sample_rate_ * duration_) * T).astype(float32)
        numpy.ndarray[float32_t, ndim=1] square_mono = (amplitude_ * numpy.sign(
            numpy.cos(omega * t_vec + phi_)) + c_).astype(float32)
        numpy.ndarray[float32_t, ndim = 2] square_stereo \
            = empty((<int>(sample_rate_ * duration_), 2), float32)

    if channel == 1:
        if format_ == -16:
            return inverse_normalize_mono(square_mono)
        else:
            return make_sound(square_mono)

    elif channel == 2:
        square_stereo[:, 0] = (amplitude_ * numpy.sign(numpy.cos(omega * t_vec + phi_)) + c_).astype(float32)
        square_stereo[:, 1] = (amplitude_ * numpy.sign(numpy.cos(omega * t_vec + phi_)) + c_).astype(float32)
        if format_ == -16:
            return inverse_normalize_stereo(square_stereo)
        else:
            return make_sound(square_stereo)

    else:
        raise ValueError(message7)


cpdef triangular_signal(float amplitude_ = 1.0,
                        float duration_  = 0.5,
                        float frequency_ = 100,
                        int sample_rate_ = 44100,
                        float ramp_ = 0.5
                        ):
    """
    CREATE A TRIANGULAR SIGNAL DATA SAMPLES
    
    :param amplitude_  : float; signal amplitude (value peak to peak)
    :param duration_   : float; Duration in seconds  
    :param frequency_  : float; Signal frequency 
    :param sample_rate_: integer; Sample rate   
    :param ramp_: float; Width of the rising ramp as a proportion of the total cycle.
        Default is 1, producing a rising ramp, while 0 produces a falling
        ramp.  `width` = 0.5 produces a triangle wave.
        If an array, causes wave shape to change over time, and must be the
        same length as t.
    :return: Return a triangular signal data samples (numpy.ndarray int16) 
    """
    if not (0.0 < amplitude_ <= 1.0) :
        raise ValueError(message8)

    if duration_ <= 0.0 :
        raise ValueError(message16)

    if frequency_ < 0.0 :
        raise ValueError(message17)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not (0 <= ramp_ <= 1.0):
        raise ValueError(message20 % "ramp")

    mixer_settings = pygame.mixer.get_init()

    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        unsigned short int channel = mixer_settings[2]
        int format_ = mixer_settings[1]
        numpy.ndarray [float32_t, ndim=1] t = numpy.linspace(0, 1, <int>(sample_rate_ * duration_)).astype(float32)

    cdef:
        numpy.ndarray[float32_t, ndim = 2] triangle_stereo \
            = empty((<int>(sample_rate_ * duration_), 2), float32)

    if channel == 1:
        triangle = numpy.asarray(amplitude_ * signal.sawtooth(
                2 * numpy.pi * frequency_ * t, ramp_), dtype=float32)
        if format_ == -16:
            return inverse_normalize_mono(triangle)
        else:
            return make_sound(triangle)

    elif channel ==2:
        triangle_stereo[:, 0] = numpy.asarray(
            amplitude_ * signal.sawtooth(2 * numpy.pi * frequency_ * t, ramp_), dtype=float32)
        triangle_stereo[:, 1] = numpy.asarray(
            amplitude_ * signal.sawtooth(2 * numpy.pi * frequency_ * t, ramp_), dtype=float32)
        if format_ == -16:
            return inverse_normalize_stereo(triangle_stereo)
        else:
            return make_sound(triangle_stereo)

    else:
        raise ValueError(message7)


cpdef cos_signal(float amplitude_ = 1.0,
                 float duration_  = 1.0,
                 float frequency_ = 100,
                 int sample_rate_ = 48000,
                 float c_         = 0.0,
                 double phi_      = 0.0
                 ):
    """
    CREATE A COSINE SIGNAL DATA SAMPLE
    
    :param amplitude_  : float; signal amplitude (normalized value [-1.0 pass 1.0]
    :param duration_   : float; duration in seconds default 1 second   
    :param frequency_  : float; frequency in hertz default 1khz
    :param sample_rate_: integer; sample rate default 48khz
    :param c_          : float; Signal constant (default zero, centered on axis y) interval [-1.0 pass 1.0]
    :param phi_        : double; Radian offset (signal phase offset) interval ]-pi pass +pi[
    :return           : return numpy ndarray shape (n, ) or (n, 2) 
    """

    if not (0.0 < fabs(amplitude_) <= 1.0) :
        raise ValueError(message8)

    if duration_ <= 0.0 :
        raise ValueError(message16)

    if frequency_ < 0.0 :
        raise ValueError(message17)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not -1.0 <= c_ <= 1.0:
        raise ValueError(message18)

    if not (fabs(phi_) < 3.1416):
        raise ValueError(message19)

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        float omega = <float>(2.0 * numpy.pi * frequency_)
        float T = <float>(1.0 / sample_rate_)
        int length = <int>(duration_ * sample_rate_)
        int i = 0
        unsigned short int channel = mixer_settings[2]
        int format_ = mixer_settings[1]
        float [::1] y_mono = empty(length, float32)
        float [:, ::1] y_stereo = empty((length, 2), float32)
        float y

    if channel == 1:
        with nogil:
            for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                y_mono[i] = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
        if format_ == -16:
            return inverse_normalize_mono(y_mono)
        else:
            return make_sound(asarray(y_mono, dtype=float32))

    elif channel == 2:
        with nogil:
            for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                y_stereo[i, 0] = y
                y_stereo[i, 1] = y
        if format_ == -16:
            return inverse_normalize_stereo(y_stereo)
        else:
            return make_sound(asarray(y_stereo, dtype=float32))
    else:
        raise NotImplementedError(message7)



cpdef cos_carrier(float amplitude_ = 1.0,
                  float duration_  = 1.0,
                  list frequencies_ = [],
                  int sample_rate_ = 44100,
                  float c_         = 0.0,
                  double phi_      = 0.0
                  ):
    """
    CREATE A CARRIER SIGNAL WITH MODULATING FREQUENCIES
    
    Create a signal corresponding to the sum of all signals determine by their respective frequency property
    
    * Output signal amplitude is within the limits [-amplitude pass +amplitude]. Modulating frequencies amplitude is 
      adjusted according to their position into the frequency list. e.g the carrier is 1/2 of the maximum amplitude, 
      the next frequency is 1/4, then 1/8 and so on.
       
    * The sum of all modulating frequency signal will always be <= abs(amplitude)
    
    :param amplitude_  : float; signal amplitude (normalized value [-1.0 pass 1.0] maximum amplitude 
    for the carrier and modulating frequencies, default 1.0
    :param duration_   : float; duration in seconds default 1 second   
    :param frequencies_: float; list of frequencies in hertz default 1khz to mix (carrier first 
    and modulating frequencies)
    :param sample_rate_: integer; sample rate default 48khz
    :param c_          : float; Signal constant (default zero, centered on axis y) interval [-1.0 pass 1.0]
    :param phi_        : double; Radian offset (signal phase offset) interval ]-pi pass +pi[
    :return           : return numpy ndarray shape (n, ) or (n, 2) 
    """

    if not (0.0 < amplitude_ <= 1.0) :
        raise ValueError(message8)

    if duration_ <= 0.0 :
        raise ValueError(message16)

    if isinstance(frequencies_, list) and len(frequencies_) == 0:
        raise ValueError(message21 % "frequencies_")

    cdef float freq = 0
    for freq in frequencies_:
        if freq < 0.0 :
            raise ValueError(message17)

    if sample_rate_ not in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    if not -1.0 <= c_ <= 1.0:
        raise ValueError(message18)

    if not (fabs(phi_) < 3.1416):
        raise ValueError(message19)

    mixer_settings = pygame.mixer.get_init()
    if mixer_settings is None:
        raise ValueError(message6)

    cdef:
        float omega = 0
        float T = <float>(1.0 / sample_rate_)
        int length = <int>(duration_ * sample_rate_)
        int i = 0
        unsigned short int channel = mixer_settings[2]
        int format_ = mixer_settings[1]
        float [::1] y_mono = zeros(length, float32)
        float [:, ::1] y_stereo = zeros((length, 2), float32)
        float y


    cdef :
        list g_mod = []
        float amp = amplitude_
        float half = 0
        int j = 0

    for freq in frequencies_:
        if freq == frequencies_[len(frequencies_)-1]:
            g_mod.append(amp)
            break
        half = amp / 2.0
        amp = amp - half
        g_mod.append(half)

    if channel == 1:

        for freq in frequencies_:
            omega = <float>(2.0 * PI * <float>freq)
            amplitude_ = g_mod[j]
            with nogil:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y_mono[i] = y_mono[i] + <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))

            j = j + 1
        if format_ == -16:
            return inverse_normalize_mono(y_mono)
        else:
            return make_sound(asarray(y_mono, dtype=float32))

    elif channel == 2:

        for freq in frequencies_:
            omega = <float>(2.0 * PI * <float>freq)
            amplitude_ = g_mod[j]
            with nogil:
                for i in prange(0, length, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    y = <float>(amplitude_ * (cos(omega * <float>i / sample_rate_ + phi_) + c_))
                    y_stereo[i, 0] = y_stereo[i, 0] + y
                    y_stereo[i, 1] = y_stereo[i, 1] + y

            j = j + 1
        if format_ == -16:
            return inverse_normalize_stereo(y_stereo)
        else:
            return make_sound(asarray(y_stereo, dtype=float32))

    else:
        raise NotImplementedError(message7)


cpdef bint is_type_memoryview(object_):
    """
    Check if the given object is a memoryview type
    
    :param object_: python object to check
    :return: bool; Return True or False 
    """
    if isinstance(object_, cython.view.memoryview):
        return True
    else:
        return False

cpdef memoryview_type_testing():
    """
    TESTING PURPOSE
    :return: return lists of memoryviews
    """
    cdef:
        short [:] mem_mono_int16_e = numpy.empty(0, int16)
        short [::1] mem_mono_int16 = numpy.empty(1024, int16)
        short [:, :] mem_stereo_int16 = numpy.empty((1024,2), int16)
        short [:, :] mem_stereo_int16_o = numpy.empty((1024,3), int16)

        float [:] mem_mono_float32_e = numpy.empty(0, float32)
        float [::1] mem_mono_float32 = numpy.empty(1024, float32)
        float [:, :] mem_stereo_float32 = numpy.empty((1024,2), float32)
        float [:, :] mem_stereo_float32_o = numpy.empty((1024,3), float32)

        double [:] mem_mono_float64_e = numpy.empty(0, float64)
        double [::1] mem_mono_float64 = numpy.empty(1024, float64)
        double [:, :] mem_stereo_float64 = numpy.empty((1024,2), float64)
        double [:, :] mem_stereo_float64_o = numpy.empty((1024,3), float64)

    l_int16 = [mem_mono_int16_e, mem_mono_int16, mem_stereo_int16, mem_stereo_int16_o]
    l_float32 = [mem_mono_float32_e, mem_mono_float32, mem_stereo_float32, mem_stereo_float32_o]
    l_float64 = [mem_mono_float64_e, mem_mono_float64, mem_stereo_float64, mem_stereo_float64_o]
    return l_int16, l_float32, l_float64

cpdef bint is_type_soundobject(object_):
    """
    Check if the given object is a soundobject type
    
    :param object_: python object to check
    :return: return True or False
    """
    if PyObject_IsInstance(object_, SOUNDTYPE):
        return True;
    else:
        return False;

cpdef bint is_type_ndarray(object_):
    """
    Check if the given object is an ndarray type
    :param object_: python object to check
    :return: return True or False
    """
    if PyObject_IsInstance(object_, numpy.ndarray):
        return True;
    else:
        return False;

cpdef bint is_valid_dtype(array_, type_=None) except *:
    """
    CHECK IF A NUMPY ARRAY DATATYPE MATCH THE DATATYPE PASSED AS ARGUMENT 
    
    :param array_: Array to check, numpy.ndarray (any shapes)
    :param type_ : Python string or list; such as 'int16' or ['int16', 'float32'] 
    :return      : True | False; True if type_ match the array datatype 
    """
    # default type_ is a list containing allowed datatypes
    if type_ is None:
        type_=['int16', 'float32']
    # otherwise type_ must be a string representing the datatype
    else:
        if not (PyObject_IsInstance(type_, str) or PyObject_IsInstance(type_, list)):
            raise ValueError(message48 % array_.dtype)

    # Check the Data-type of the array’s elements.
    if PyObject_HasAttr(array_, 'dtype'):
        if PyObject_IsInstance(type_, list):
            if array_.dtype in type_:
                return True

        elif PyObject_IsInstance(type_, str):
                if array_.dtype == type_:
                    return True

    return False


cpdef bint is_valid_array(array_):
    """
    CHECK IF GIVEN ARRAY (MONO OR STEREO) IS VALID OR NOT (ARRAY TYPE IS NOT CHECKED, ONLY SHAPES)
    
    :param array_: sndarray; Mono or stereo sndarray representing sound samples 
    :return: Return a Cython object bint True | False 
    """
    if PyObject_HasAttr(array_, 'shape'):
        # mono
        if len(array_.shape) == 1:
            return True

        # stereo
        elif len(array_.shape) == 2:
            if array_.shape[1] == 2:
                return True # stereo sound

    return False


cpdef bint is_valid_mono_array(array_):
    """
    DETERMINE IF ARRAY SAMPLES (MONO) IS VALID OR NOT (samples, ) 
    
    * Only Array type int16 and float32 are compatible any other types will return False 
     
    :param array_: sndarray; Mono sndarray representing sound samples 
    :return: Return a Cython object (bint) True | False 
    """
    if PyObject_HasAttr(array_, 'shape'):
        if len(array_.shape) == 1:
            if isinstance(array_, cython.view.memoryview):
                if is_valid_dtype(numpy.asarray(array_)):
                    return True
            else:
                if is_valid_dtype(array_):
                    return True  # mono sound
    return False


cpdef bint is_valid_stereo_array(array_):
    """
    DETERMINE IF ARRAY SAMPLES (STEREO) IS A VALID OR NOT (samples, 2)
    
    * Only Array type int16 or float32 are compatible any other types will return False
     
    :param array_: sndarray; Mono sndarray representing sound samples 
    :return: Return a Cython object (bint) True | False 
    """
    if PyObject_HasAttr(array_, 'shape'):
        if len(array_.shape) == 2:
            if array_.shape[1] == 2:
                if isinstance(array_, cython.view.memoryview):
                    if is_valid_dtype(numpy.asarray(array_))  :
                        return True
                else:
                    if is_valid_dtype(array_):
                        return True  # mono sound
    return False


cpdef bint is_monophonic(array_):
    """
    CHECK IF A SOUND ARRAY IS MONOPHONIC (ONLY COMPATIBLE WITH FLOAT DATATYPE)
    
    * Input: Array_ must be a numpy.ndarray type float_, float32 or float64
    
    :param array_: ndarray; data samples (array must be type float_, float32 or float64)
    :return: True | False 
    """
    if PyObject_HasAttr(array_, 'shape'):
        # monophonic
        if len(array_.shape) == 1:
            if isinstance(array_, cython.view.memoryview):
                if numpy.asarray(array_).dtype in (float_, float32, float64):
                    return True
            elif isinstance(array_, numpy.ndarray):
                if array_.dtype in (float_, float32, float64):
                    return True

    return False

cpdef bint is_stereophonic(array_):
    """
    CHECK IF A SOUND ARRAY IS STEREOPHONIC (ONLY COMPATIBLE WITH FLOAT DATATYPE)
    
    * Input: Array_ must be a numpy.ndarray type float_, float32 or float64
    
    :param array_: ndarray; data samples (array must be type float_, float32 or float64)
    :return: True | False 
    """
    if PyObject_HasAttr(array_, 'shape'):
        if isinstance(array_, cython.view.memoryview):
            arr = numpy.asarray(array_)
            if len(arr.shape) == 2 and arr.shape[1] == 2:
                if arr.dtype in (float_, float32, float64):
                    return True

        elif isinstance(array_, numpy.ndarray):
            if len(array_.shape) == 2 and array_.shape[1] == 2:
                if array_.dtype in (float_, float32, float64):
                    return True
    return False

cpdef float sound_length(sound_, int fs_) except *:
    """
    RETURN SOUND LENGTH IN SECONDS 
    
    Sound can be a pygame.mixer.Sound object or a numpy ndarray (datatype int16 or float32) representing a mono 
    or stereo sound 
    
    :param sound_: pygame.Sound | ndarray (datatype int16 or float32); pygame.mixer.Sound mono or stereo 
    :param fs_   : float; sample rate
    :return      : float; return sound length in seconds. 
    """
    # Check the sample rate
    if not (fs_ in FS):
        raise ValueError(message15 % (fs_, FS))

    if is_type_soundobject(sound_):
        return <float>(pygame.sndarray.samples(sound_).shape[0] / <float>fs_)

    elif is_type_ndarray(sound_):
        if is_valid_array(sound_):
            if is_valid_dtype(sound_):
                return <float>(sound_.shape[0] / <float>fs_)

    elif isinstance(sound_, cython.view.memoryview):
        arr = numpy.asarray(sound_)
        if is_valid_array(sound_):
            print(arr.dtype)
            if arr.dtype in (int16, float32):
                return <float>(sound_.shape[0] / <float>fs_)
    raise ValueError(message30)


cpdef dict sound_values(sound_):
    """
    RETURN PYTHON DICTIONARY WITH MIN, AVG & MAX VALUES OF A PYGAME SOUND
     
    :param sound_ : pygame.Sound; Mono or stereo pygame.Sound object 
    :return       : Return a python dict containing min, avg and max values  
    """
    cdef:
        dict c1, c2

    if not is_type_soundobject(sound_):
        raise ValueError(message39)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if is_valid_mono_array(sound_array):
        pass
    elif is_valid_stereo_array(sound_array):
        pass
    else:
        raise ValueError(message30)

    # mono
    if len(sound_array.shape) == 1:
        return {
                "min":round(<float>amin(sound_array[:]), 4),
                "avg":round(<float>average(sound_array[:]), 4),
                "max":round(<float>amax(sound_array[:]), 4)
        }
    # stereo
    elif len(sound_array.shape) == 2:

        c1 = {
            "min":round(<float>amin(sound_array[:, 0]), 4),
            "avg":round(<float>average(sound_array[:, 0]), 4),
            "max":round(<float>amax(sound_array[:, 0]), 4)
        }

        c2 = {"min":round(<float>amin(sound_array[:, 1]), 4),
              "avg":round(<float>average(sound_array[:, 1]), 4),
              "max":round(<float>amax(sound_array[:, 1]), 4)
        }
    else:
        raise ValueError(message30)

    return {"channel0" : c1, "channel1" : c2}



cpdef void display_sound_values(sound_) except *:
    """
    DISPLAY SOUND MIN, AVG, MAX VALUES (MONO AND STEREO)
    
    :param sound_: pygame.Sound; Mono or stereo pygame.Sound object 
    :return: Void
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message39)

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if is_valid_mono_array(sound_array):
        pass
    elif is_valid_stereo_array(sound_array):
        pass
    else:
        raise ValueError(message30)

    if len(sound_array.shape) == 1:
         printf("\nMonophonic min:{min:9f} avg:{average:5f} max:{max:9f}".format(
             min     = round(amin(sound_array), 4),
             average = round(average(sound_array), 4),
             max     = round(amax(sound_array), 4)).encode('utf-8'))

    elif len(sound_array.shape) == 2:
        printf("\nChannel0 (min={min0:9f} avg={average0:9f} max={max0:9f}), " \
        "\nchannel1 (min={min1:9f} avg={average1:9f} max={max1:9f}) ".format(
            min0    = round(amin(sound_array[:, 0]), 4),
            average0= round(average(sound_array[:, 0]), 4),
            max0    = round(amax(sound_array[:, 0]), 4),
            min1    = round(amin(sound_array[:, 1]), 4),
            average1= round(average(sound_array[:, 1]), 4),
            max1    = round(amax(sound_array[:, 1]), 4)).encode('utf-8'))
    else:
        raise ValueError(message30)


cpdef float rms_value_mono(sound_) except *:
    """
    RETURN RMS VALUE OF A MONOPHONIC SOUND (RMS VALUES IN DECIBELS)
    
    :param sound_: pygame Sound; The sound must be a monophonic sound  
    :return       : float; Returns RMS scalar value
    """
    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if is_valid_mono_array(sound_array):
        pass
    else:
        raise ValueError(message11)

    if sound_array.dtype == int16:
        sound_array = normalize_array_mono(sound_array)

    elif sound_array.dtype == float32:
        pass
    else:
        raise ValueError(message27 % sound_array.dtype)

    cdef:
        float [::1] samples_norm = sound_array
        int n0 = <object>sound_array.shape[0]

    assert n0 != 0, message12

    cdef float centre   = <float>(10.0 * log10(numpy.sum(numpy.square(samples_norm)) / <float>n0))
    return centre

cpdef rms_values_stereo(sound_):
    """
    RETURN RMS VALUES FOR A GIVEN STEREO SOUND (RETURN LEFT, RIGHT CENTRE RMS VALUES)
    
    :param sound_: pygame Sound; The sound must be a stereophonic sound  
    :return       : Return a python tuple of scalars values (double) representing the RMS
                    values of the left, right and centre channels
    """
    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if is_valid_stereo_array(sound_array):
        pass
    else:
        raise ValueError(message14)

    if sound_array.dtype == int16:
        sound_array = normalize_array_stereo(sound_array)
    elif sound_array.dtype == float32:
        pass
    else:
        raise ValueError(message27 % sound_array.dtype)

    cdef:
        float [:, :] samples_norm = sound_array
        int n0 = <object>sound_array.shape[0]

    assert n0 != 0, message12

    cdef :
        int n = 2 * n0
        float rms_left   = <float>(10.0 * log10(numpy.sum(numpy.square(samples_norm[:, 0])) / <float>n0))
        float rms_right  = <float>(10.0 * log10(numpy.sum(numpy.square(samples_norm[:, 1])) / <float>n0))
        float rms_stereo = <float>(10.0 * log10(numpy.sum(numpy.square(samples_norm[:, :])) / <float>n))

    return  rms_left, rms_right, rms_stereo


cpdef void show_rms_values(sound):
    """
    DISPLAY RMS VALUES FOR THE GIVEN SOUND 
    
    :param sound: pygame Sound; Sound (stereophonic or monophonic sound) 
    :return: Void
    """
    try:
        samples = pygame.sndarray.samples(sound)
    except:
        raise ValueError(message39)

    if is_valid_mono_array(samples):
        if samples.dtype == int16:
            samples = normalize_array_mono(samples)

    elif is_valid_stereo_array(samples):
        if samples.dtype == int16:
            samples = normalize_array_stereo(samples)
    else:
       raise ValueError(message30)

    cdef int channel_number = len(samples.shape)

    cdef:
        int n0 = <object>samples.shape[0]
        float [:, :] norm_stereo = samples if channel_number==2 else empty((n0, 2), float32)
        float [::1] norm_mono = samples if channel_number==1 else empty(n0, float32)
        int n = 2 * n0
        float rms_left, rms_right, rms_stereo, rms_center

    if n0 == 0:
        raise ValueError(message12)

    if channel_number == 1:
        rms_center = <float>(10.0 * log10(numpy.sum(numpy.square(norm_mono)) / <float>n0))
        printf("\nrms={centre:8f})".format(centre=round(rms_center, 4)).encode('utf-8'))

    elif channel_number == 2:
        rms_left   = <float>(10.0 * log10(numpy.sum(numpy.square(norm_stereo[:, 0])) / <float>n0))
        rms_right  = <float>(10.0 * log10(numpy.sum(numpy.square(norm_stereo[:, 1])) / <float>n0))
        rms_stereo = <float>(10.0 * log10(numpy.sum(numpy.square(norm_stereo[:, :])) / <float>n))
        printf("\nleft={left:8f} right={right:8f} "
               "centre={centre:8f})".format(left=round(rms_left, 4),
                right=round(rms_right, 4), centre=round(rms_stereo, 4)).encode('utf-8'))
    else:
        raise ValueError(message30)


cpdef inverse_normalize_mono(float [:] samples):
    """
    TAKE AN ARRAY FLOAT AS INPUT (SOUND SAMPLES FLOAT32) AND RETURN A PYGAME.MIXER.SOUND 
    
    :param samples: ndarray; Take a 2d numpy.ndarray type float32 (stereo)  
    :return       : return a pygame.sound object ready to be play on the mixer
    """
    if not is_monophonic(samples):
        raise ValueError(message11)

    cdef:
        int width = <object>samples.shape[0]
        int i
        float s0, s1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples[i]
            if s0 > 0:
                samples[i] = s0 * SHRT_MAX
            elif s0 < 0:
                samples[i] = -s0 * SHRT_MIN
            else:
                samples[i] = 0.0
    return make_sound(asarray(samples, dtype=int16))


cpdef inverse_normalize_stereo(float [:, :] samples):
    """
    TAKE AN ARRAY float AS INPUT (SOUND SAMPLES) AND RETURN A PYGAME.MIXER.SOUND 
    
    :param samples: ndarray; Take a 2d numpy.ndarray type float (stereo)  
    :return       : return a pygame.sound object ready to be play on the mixer
    """

    if not is_stereophonic(samples):
        raise ValueError(message11)

    cdef:
        int width = <object>samples.shape[0]
        int i
        float s0, s1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples[i, 0]
            s1 = samples[i, 1]
            if s0 > 0:
                samples[i, 0] = s0 * SHRT_MAX
            elif s0 < 0:
                samples[i, 0] = -s0 * SHRT_MIN
            else:
                samples[i, 0] = 0.0

            if s1 > 0:
                samples[i, 1] = s1 * SHRT_MAX
            elif s1 < 0:
                samples[i, 1] = -s1 * SHRT_MIN
            else:
                samples[i, 1] = 0.0

    return make_sound(asarray(samples, dtype=int16))


cpdef inverse_normalize_stereo_asarray(float [:, :] samples_):
    """
    TAKE AN ARRAY AS INPUT (SOUND SAMPLES) AND RETURN A NUMPY ARRAY INT16
    
    :param samples_: ndarray; Take a 2d numpy.ndarray type float (stereo)  
    :return       : numpy.ndarray : Return a array shape (n, 2) int16
    """
    if not is_stereophonic(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_.shape[0]
        int i
        float s0, s1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples_[i, 0]
            s1 = samples_[i, 1]
            if s0 > 0:
                samples_[i, 0] = s0 * SHRT_MAX
            elif s0 < 0:
                samples_[i, 0] = -s0 * SHRT_MIN
            else:
                samples_[i, 0] = 0.0

            if s1 > 0:
                samples_[i, 1] = s1 * SHRT_MAX
            elif s1 < 0:
                samples_[i, 1] = -s1 * SHRT_MIN
            else:
                samples_[i, 1] = 0.0

    return asarray(samples_, dtype=int16)


cpdef inverse_normalize_mono_asarray(float [:] samples_):
    """
    TAKE AN ARRAY float AS INPUT (SOUND SAMPLES) AND RETURN A NUMPY ARRAY INT16 VALUES
    
    :param samples_: ndarray; Take a 2d numpy.ndarray type float (stereo)  
    :return       : numpy.ndarray; array shape (n, ) int16
    """
    if not is_monophonic(samples_):
        raise ValueError(message11)

    cdef:
        int width = <object>samples_.shape[0]
        int i
        float s0, s1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            s0 = samples_[i]
            if s0 > 0:
                samples_[i] = s0 * SHRT_MAX
            elif s0 < 0:
                samples_[i] = -s0 * SHRT_MIN
            else:
                samples_[i] = 0.0

    return asarray(samples_, dtype=int16)


cpdef time_shift_mono_int16(short [:] samples_, float shift_, int sample_rate_):
    """
    SHIFT SAMPLES DATA TO A GIVEN AMOUNT OF TIME IN MS (COMPATIBLE MONO ONLY)
    
    :param sample_rate_: int; python int that represents the sample rate 
    :param samples_    : numpy.sndarray; data samples, must be a valid mono type array filled with int16 values
    :param shift_      : float; number of milli-seconds 
    :return            : pygame.mixer.Sound; Return a Sound with shifted channels effect
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    if not sample_rate_ in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float max_length = <float>width/<float>sample_rate_ * 1000

    if not (0.0 <= shift_ <= max_length):
        raise ValueError(message45 % (0, shift_, max_length, shift_))

    cdef:
        short [::1] new_array = zeros(width, int16)
        int i, l
        int s = <int>((shift_ / 1000.0) * sample_rate_)

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            l = i - s
            if l < 0: l = 0
            new_array[i] = <short>samples_[l]

    return make_sound(asarray(new_array))



cpdef time_shift_mono_float32(float [:] samples_, float shift_, int sample_rate_):
    """
    SHIFT SAMPLES DATA TO A GIVEN AMOUNT OF TIME IN MS (COMPATIBLE MONO FLOAT32 ONLY)
    
    :param sample_rate_: int; python int that represents the sample rate
    :param samples_    : numpy.sndarray; data samples, must be a valid mono type array filled with float32 values
    :param shift_      : float; number of milli-seconds 
    :return            : pygame.mixer.Sound; Return a Sound with shifted channels effect
    """

    if not is_monophonic(samples_):
        raise ValueError(message11)

    if not sample_rate_ in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float max_length = <float>width/<float>sample_rate_ * 1000

    if not (0.0 <= shift_ <= max_length):
        raise ValueError(message45 % (0, shift_, max_length, shift_))

    cdef:
        float [::1] new_array = zeros(width, float32)
        int i, l
        int s = <int>((shift_ / 1000.0) * sample_rate_)

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            l = i - s
            if l < 0: l = 0
            new_array[i] = <float>samples_[l]

    return make_sound(asarray(new_array))


cpdef time_shift_stereo_int16(short [:, :] samples_, float shift_, int sample_rate_):
    """
    SHIFT SAMPLES DATA TO A GIVEN AMOUNT OF TIME IN MS (BOTH CHANNELS) 
    
    :param sample_rate_: int; python int that represents the sample rate 
    :param samples_    : numpy.sndarray; data samples, must be a valid stereo type array filled with int16 values
    :param shift_      : float; number of milli-seconds 
    :return            : pygame.mixer.Sound; Return a Sound with shifted channels effect
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if not sample_rate_ in FS:
        raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float max_length = <float>width/<float>sample_rate_ * 1000

    if not (0.0 <= shift_ <= max_length):
        raise ValueError(message45 % (0, shift_, max_length, shift_))

    cdef:
        short [:, :] new_array = zeros((width, 2), int16)
        int i, l
        int s = <int>((shift_ / 1000.0) * sample_rate_)

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            l = i - s
            if l < 0: l = 0
            new_array[i, 0] = <short>samples_[l, 0]
            new_array[i, 1] = <short>samples_[l, 1]

    return make_sound(asarray(new_array, dtype=int16))


cpdef time_shift_stereo_float32(float [:, :] samples_, float shift_, int sample_rate_):
    """
    SHIFT SAMPLES DATA TO A GIVEN AMOUNT OF TIME IN MS (BOTH CHANNELS) 
    
    :param sample_rate_: int; python int that represents the sample rate 
    :param samples_    : numpy.sndarray; data samples, must be a valid stereo type array filled with float32 values
    :param shift_      : float; number of milli-seconds 
    :return            : pygame.mixer.Sound; Return a Sound with shifted channels effect
    """

    if not is_stereophonic(samples_):
        raise ValueError(message14)

    if not sample_rate_ in FS:
         raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float max_length = <float>width/<float>sample_rate_ * 1000

    if not (0.0 <= shift_ <= max_length):
        raise ValueError(message45 % (0, shift_, max_length, shift_))

    cdef:

        float [:, :] new_array = zeros((width, 2), float32)
        int i, l
        int s = <int>((shift_ / 1000.0) * sample_rate_)

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            l = i - s
            if l < 0: l = 0
            new_array[i, 0] = <float>samples_[l, 0]
            new_array[i, 1] = <float>samples_[l, 1]

    return make_sound(asarray(new_array))


cpdef time_shift_channel(short [:, :] samples_, float shift_, int sample_rate_, unsigned short int channel_=0):
    """
    SHIFT A GIVEN CHANNEL OF A STEREO ARRAY 
    
    :param channel_    : int; python int (channel to shift). Default channel zero
    :param sample_rate_: int; python int that represents the sample rate 
    :param samples_    : numpy.sndarray; data samples, must be a valid stereo type array filled with int16 values
    :param shift_      : float; number of milli-seconds 
    :return            : pygame.mixer.Sound; Return a Sound with a single channel shifted by (n ms)
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if not sample_rate_ in FS:
         raise ValueError(message15 % (sample_rate_, FS))

    cdef:
        int width = <object>samples_.shape[0]
        float max_length = <float>width/<float>sample_rate_ * 1000

    if width == 0:
        raise ValueError(message12)

    if not (0.0 <= shift_ <= max_length):
        raise ValueError(message45 % (0, shift_, max_length, shift_))

    if channel_ not in [0, 1]:
        raise ValueError(message7)

    cdef:
        short [:, :] new_array = zeros((width, 2), int16)
        int i, l
        int s = <int>((shift_ / 1000.0) * sample_rate_)

    with nogil:
        if channel_ == 0:
            for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                l = i - s
                if l < 0: l = 0
                new_array[i, 0] = <short>samples_[l, 0]
                new_array[l, 1] = <short>samples_[l, 1]
        else:
            for i in prange(s, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                l = i - s
                if l < 0: l = 0
                new_array[l, 0] = <short>samples_[l, 0]
                new_array[i, 1] = <short>samples_[l, 1]

    return make_sound(asarray(new_array))


cpdef void set_volume(sound_, float volume_=1.0) except *:
    """
    SET THE PLAYBACK VOLUME (INPLACE)
    
    :param sound_ : pygame.mixer.Sound object (stereo or mono) 
    :param volume_: float; volume must be in range [0.0 pass 1.0], 
                    volume out of range will be reverted to default value 1.0
                    max volume is 1.0 and lowest 0.0
    :return       : void (sound volume changed inplace)
    """

    if volume_ < 0.0:
        volume_ = 0.0

    # The sound sample will remains un-touched
    if volume_ > 1.0:
        pass

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    try:
        sound_array  = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if is_valid_stereo_array(sound_array):
        pass
    elif is_valid_mono_array(sound_array):
        pass
    else:
        raise ValueError(message30)

    cdef:
        int width          = sound_array.shape[0]
        int channel_number = len(sound_array.shape)
        int i
        short [::1] array_mono_int16 =  \
            sound_array if (channel_number == 1 and sound_array.dtype==int16) else empty(width, int16)
        float [::1] array_mono_float32 = \
            sound_array if (channel_number == 1 and sound_array.dtype==float32) else empty(width, float32)
        short [:, :] array_stereo_int16 = \
            sound_array if (channel_number == 2 and sound_array.dtype==int16) else empty((width, 2), int16)
        float [:, :] array_stereo_float32 = \
            sound_array if (channel_number == 2 and sound_array.dtype==float32) else empty((width, 2), float32)
    if width == 0:
        raise ValueError(message12)

    if channel_number == 2:
        if sound_array.dtype == float32:
            with nogil:
                for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    array_stereo_float32[i, 0], array_stereo_float32[i, 1] = \
                        array_stereo_float32[i, 0] * volume_, array_stereo_float32[i, 1] * volume_
        else:
            with nogil:
                for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    array_stereo_int16[i, 0], array_stereo_int16[i, 1] = \
                        <short>(array_stereo_int16[i, 0] * volume_), <short>(array_stereo_int16[i, 1] * volume_)

    elif channel_number == 1:
        if sound_array.dtype == float32:
            with nogil:
                for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    array_mono_float32[i] = array_mono_float32[i] * volume_
        else:
            with nogil:
                for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    array_mono_int16[i] = <short>(array_mono_int16[i] * volume_)
    else:
        raise ValueError(message30)


# TODO NEED TESTING AND CHECK PROFILING PERFORMANCES
cpdef reverse_sound_beta(sound_):
    """
    REVERSE A SOUND (MAKE A SOUND PLAY BACKWARDS)
    
    :param sound_: pygame Sound; monophonic or stereophonic sound to play backwards
    :return: pygame Sound; return a sound playing backwards 
    """
    try:
        sound_array = pygame.sndarray.array(sound_)
    except:
        raise ValueError(message39)

    cdef:
        int width = sound_array.shape[0]

    if width == 0:
        raise ValueError(message12)

    if is_valid_stereo_array(sound_array):
        sound_array_ = numpy.ascontiguousarray(sound_array[::-1, :])

    elif is_valid_mono_array(sound_array):
        sound_array_ = numpy.ascontiguousarray(sound_array[::-1])

    else:
        raise ValueError(message30)

    return make_sound(sound_array_)


cpdef reverse_stereo_int16(short [:, :] samples_):
    """
    REVERSE SOUND EFFECT (PLAY BACKWARD)
    
    :param samples_: numpy.ndarray; Numpy.ndarray int16 values stereo representing a sound effect
    :return        : pygame.mixer.Sound; Return a sound object.
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_[:, 0].shape[0]
        short [:, :] new_array = zeros((width, 2), dtype=int16)
        int i = 0, j =0

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width-1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            j = width - i
            new_array[i, 0] = samples_[j, 0]
            new_array[i, 1] = samples_[j, 1]

    return make_sound(asarray(new_array))

cpdef reverse_stereo_float32(float [:, :] samples_):
    """
    REVERSE SOUND EFFECT  
    
    :param samples_: numpy.ndarray; Numpy.ndarray float32 values stereo representing a sound effect
    :return        : pygame.mixer.Sound; Return a sound object.
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_[:, 0].shape[0]
        float [:, :] new_array = zeros((width, 2), dtype=float32)
        int i = 0, j =0

    with nogil:
        for i in prange(width-1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            j = width - i
            new_array[i, 0] = samples_[j, 0]
            new_array[i, 1] = samples_[j, 1]

    return make_sound(asarray(new_array))

cpdef reverse_mono_int16(short [:] samples_):
    """
    REVERSE SOUND EFFECT
    
    :param samples_: numpy.ndarray; Numpy.ndarray int16 values mono representing a sound effect
    :return       : pygame.mixer.Sound; Return a sound object.
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    cdef:
        int width = <object>samples_.shape[0]
        short [:] new_array = zeros(width, dtype=int16)
        int i = 0, j =0

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            j = width - i
            new_array[i] = samples_[j]

    return make_sound(asarray(new_array))

cpdef reverse_mono_float32(float [:] samples_):
    """
    REVERSE SOUND EFFECT
    
    :param samples_: numpy.ndarray; Numpy.ndarray float32 values mono representing a sound effect
    :return       : pygame.mixer.Sound; Return a sound object.
    """
    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    cdef:
        int width = <object>samples_.shape[0]
        float [:] new_array = zeros(width, dtype=float32)
        int i = 0, j =0

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            j = width - i
            new_array[i] = samples_[j]

    return make_sound(asarray(new_array, dtype=int16))


cpdef reverse_sound(sound_):
    """
    REVERSE A SOUND (MONOPHONIC OR STEREOPHONIC)
        
    :param sound_: pygame sound; Pygame sound effect to reverse
    :return: return a pygame sound effect reversed (playing backward)
    """

    try:
        sound_array = pygame.sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    cdef:
        int width = sound_array.shape[0]
        int channel_number = len(sound_array.shape)

    if width == 0:
        raise ValueError(message12)

    if is_valid_mono_array(sound_array):
        pass
    elif is_valid_stereo_array(sound_array):
        pass
    else:
        raise ValueError(message30)

    cdef:
        short [::1] array_mono_int16 =  \
            sound_array if (channel_number == 1 and sound_array.dtype==int16) else empty(width, int16)
        float [::1] array_mono_float32 = \
            sound_array if (channel_number == 1 and sound_array.dtype==float32) else empty(width, float32)
        short [:, :] array_stereo_int16 = \
            sound_array if (channel_number == 2 and sound_array.dtype==int16) else empty((width, 2), int16)
        float [:, :] array_stereo_float32 = \
            sound_array if (channel_number == 2 and sound_array.dtype==float32) else empty((width, 2), float32)
        int i = 0, j =0
        short [::1] new_array_mono_int16 = empty(width, int16)
        float [::1] new_array_mono_float32 = empty(width, float32)
        short [:, :] new_array_stereo_int16 = empty((width, 2), int16)
        float [:, :] new_array_stereo_float32 = empty((width, 2), float32)

    if channel_number == 1:
        if sound_array.dtype == numpy.int16:
            with nogil:
                for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    j = width - i
                    new_array_mono_int16[i] = array_mono_int16[j]
            return make_sound(asarray(new_array_mono_int16))

        elif sound_array.dtype == numpy.float32:
            with nogil:
                for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    j = width - i
                    new_array_mono_float32[i] = array_mono_float32[j]
            return make_sound(asarray(new_array_mono_float32))

    elif channel_number == 2:
        if sound_array.dtype == numpy.int16:
            with nogil:
                for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    j = width - i
                    new_array_stereo_int16[i] = array_stereo_int16[j]
            return make_sound(asarray(new_array_stereo_int16))

        elif sound_array.dtype == numpy.float32:
            with nogil:
                for i in prange(width - 1, 0, -1, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    j = width - i
                    new_array_stereo_float32[i] = array_stereo_float32[j]
            return make_sound(asarray(new_array_stereo_float32))

    else:
        raise ValueError(message7)


cpdef invert_array_mono_int16(short [:] samples_):
    """
    INVERT A SOUND EFFECT (MONOPHONIC)
    
    :param samples_: numpy.ndarray; Array shape (n, ) int16 representing a monophonic sound effect 
    :return: Return a pygame Sound (inverted sound effect)
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    cdef:
        int width = <object>samples_.shape[0]
        int i     = 0

    if width == 0:
        raise ValueError(message12)

    with nogil:

        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):

            if samples_[i] > 0:
                samples_[i] = <short>(-samples_[i] - 1)
            else:
                samples_[i] = <short>(abs(samples_[i]) - 1)

    return make_sound(asarray(samples_))


cpdef invert_array_mono_float32(float [:] samples_):
    """
    INVERT A SOUND EFFECT (MONOPHONIC)
    
    :param samples_: numpy.ndarray; Array shape (n, ) float32 representing a monophonic sound effect 
    :return: Return a pygame Sound (inverted sound effect)
    """

    if not is_valid_mono_array(samples_):
        raise ValueError(message11)

    cdef:
        int width = <object>samples_.shape[0]
        int i     = 0

    if width == 0:
        raise ValueError(message12)

    with nogil:

        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
             samples_[i] = -samples_[i]
    return make_sound(asarray(samples_))


cpdef invert_array_stereo_int16(short [:, :] samples_):
    """
    INVERT A SOUND EFFECT (STEREOPHONIC)
    
    :param samples_: numpy.ndarray; Array shape (n, 2) int16 representing a stereophonic sound effect 
    :return: Return a pygame Sound (inverted sound effect)
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_.shape[0]
        int i     = 0
        short c1, c2

    if width == 0:
        raise ValueError(message12)

    with nogil:

        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            c1 = samples_[i, 0]  # channel 0
            c2 = samples_[i, 1]  # channel 1
            if c1 > 0:
                samples_[i, 0] = -c1 - 1
            else:
                samples_[i, 0] = abs(c1) - 1

            if c2 > 0:
                samples_[i, 1] = -c2 - 1
            else:
                samples_[i, 1] = abs(c2) - 1

    return make_sound(asarray(samples_))


cpdef invert_array_stereo_float32(float [:, :] samples_):
    """
    INVERT A SOUND EFFECT (STEREOPHONIC)
    
    :param samples_: numpy.ndarray; Array shape (n, 2) float32 representing a stereophonic sound effect 
    :return: Return a pygame Sound (inverted sound effect)
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int width = <object>samples_.shape[0]
        int i     = 0
        short c1, c2

    if width == 0:
        raise ValueError(message12)

    with nogil:

        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            samples_[i, 0] = -samples_[i, 0]
            samples_[i, 1] = -samples_[i, 1]

    return make_sound(asarray(samples_))


cpdef void adaptative_g_stereo_float(float [:, :] array_):
    """
    Increase data sample signal amplitude by a factor f given by 1.0 / max_value. 
    The max value correspond to the maximum value within the array. If the maximum 
    value in the data sample is 0.5, f will be 2.0 and the signal amplitude will 
    be x2 and still remain in the range [-1.0 ... 1.0] without affecting the frequency domain 
    or phase.
    
    :param array_: numpy.ndarray; array shape (n, 2) float32 
    :return: void; operation inplace
    """

    cdef:
        float max_v = 0.0, f = 1.0
        int i = 0
        int width = len(<object>array_[:, 0])



    with nogil:
        max_v = f_max(&array_[0, 0], width)

        # silence or empty data sample will have max value = 0
        # Return without changing the array.
        if max_v == 0:
            return

        if max_v < 1.0:
            f = 1.0 / max_v
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                array_[i, 0] *= f
                array_[i, 1] *= f


cpdef void adaptative_g_mono_float(float [:] array_):
    """
    Increase data sample signal amplitude by a factor f given by 1.0 / max_value. 
    The max value correspond to the maximum value within the array. If the maximum 
    value in the data sample is 0.5, f will be 2.0 and the signal amplitude will 
    be x2 and still remain in the range [-1.0 ... 1.0] without affecting the frequency domain 
    or phase.
    
    :param array_: numpy.ndarray; array shape (n,) float32 
    :return: void; operation inplace
    """

    cdef:
        float max_v = 0.0, f = 1.0
        int i = 0
        int width = len(<object>array_)

    with nogil:
        max_v = f_max(&array_[0], width)
        # silence or empty data sample will have max value = 0
        # Return without changing the array.
        if max_v == 0:
            return
        if max_v < 1.0:
            f = 1.0 / max_v
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                array_[i] *= f


cpdef void adaptative_g_stereo_int16(short [:, :] array_):
    """
    Increase data sample signal amplitude by a factor f given by 1.0 / max_value.
    The max value correspond to the maximum value within the array. If the maximum
    value in the data sample is 0.5, f will be 2.0 and the signal amplitude will
    be x2 and still remain in the range [SHRT_MIN... SHRT_MAX] without affecting the frequency domain
    or phase.

    :param array_: numpy.ndarray; array shape (n, 2) int16
    :return: void; operation inplace
    """
    cdef:
        float max_v = 0, v1, v2, f = 1.0
        short f_pos = SHRT_MAX  # 32767
        short f_neg = SHRT_MIN # -32767 - 1
        int i = 0
        int width = len(<object>array_[:, 0])
        float [:, :] tmp_array = normalize_array_stereo(array_)

    with nogil:
        max_v = f_max(&tmp_array[0, 0], width)

        # silence or empty data sample will have max value = 0
        # Return without changing the array.
        if max_v == 0:
            return

        if max_v < 1.0:
            f = 1.0 / max_v
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                v1 =  tmp_array[i, 0] * f
                v2 =  tmp_array[i, 1] * f
                if v1 > 0:
                    v1 = v1 * <float>f_pos
                else:
                    v1 = -v1 * <float>f_neg
                if v2 > 0:
                    v2 = v2 * <float>f_pos
                else:
                    v2 = -v2 * <float>f_neg
                array_[i, 0] = <short>v1
                array_[i, 1] = <short>v2



cpdef void adaptative_g_mono_int16(short [:] array_):
    """
    Increase data sample signal amplitude by a factor f given by 1.0 / max_value. 
    The max value correspond to the maximum value within the array. If the maximum 
    value in the data sample is 0.5, f will be 2.0 and the signal amplitude will 
    be x2 and still remain in the range [SHRT_MIN... SHRT_MAX] without affecting the frequency domain 
    or phase.
    
    :param array_: numpy.ndarray; array shape (n,) int16 
    :return: void; operation inplace
    """

    cdef:
        float max_v = 0, v, f = 1.0
        short f_pos = SHRT_MAX  # 32767
        short f_neg = SHRT_MIN # -32767 - 1
        int i = 0
        int width = len(<object>array_)
        float [:] tmp_array = normalize_array_mono(array_)

    with nogil:
        # todo max_v cannot be zero
        max_v = f_max(&tmp_array[0], width)

        # silence or empty data sample will have max value = 0
        # Return without changing the array.
        if max_v == 0:
            return

        if max_v < 1.0:
            f = 1.0 / max_v
            for i in prange(0, width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                v =  tmp_array[i] * f
                if v > 0:
                    v = v * <float>f_pos
                else:
                    v = -v * <float>f_neg
                array_[i] = <short>v


cpdef adding_stereo_int16(short [:, :] sound0, short[:, :] sound1, bint set_gain_ = False):
    """
    MIX TWO STEREOPHONIC SOUNDS TOGETHER (COMPATIBLE INT16)
    
    * Data samples can have different sizes, the final sound will be the combination of 
    both sounds but the final sound length will be equal to the shortest sound.
    * Both data samples values are multiply by 1/2 before the mix to cap the values 
    
    * Compatible with stereophonic sounds int16
    
    :param sound0: numpy.ndarray; First sound (stereophonic) 2d numpy array shape (n, 2) int16 datatype
    :param sound1: numpy.ndarray; Second sound to mix (stereophonic). 2d numpy array shape (m, 2) int16
    :param set_gain_: bool; set the gain (increase source data samples values). When set to True the data 
    samples values are multiply par a factor f corresponding to the inverse of the maximum value f=1.0/max_value.
    Both sound data samples will received a proportional volume boost.
    :return : pygame.mixer.Sound; Return a pygame Sound with length equal to the shortest sounds passed as 
    argument to the method.
    """

    if not is_valid_stereo_array(sound0):
        raise ValueError(message26 % 1)

    if not is_valid_stereo_array(sound1):
        raise ValueError(message26 % 2)

    cdef:
        int width0 = <object>sound0.shape[0]
        int width1 = <object>sound1.shape[0]
        int minimum = min(width0, width1)
        int i = 0
        short [:, :] new_array = empty((minimum, 2), int16)

    if minimum == 0:
        raise ValueError(message12)

    if set_gain_:
        adaptative_g_stereo_int16(sound0)
        adaptative_g_stereo_int16(sound1)

    with nogil:
        for i in prange(0, minimum, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_array[i, 0] = <short>((sound0[i, 0] + sound1[i, 0]) / 2.0)
            new_array[i, 1] = <short>((sound0[i, 1] + sound1[i, 1]) / 2.0)

    return make_sound(asarray(new_array))

cpdef adding_stereo_float32(float [:, :] sound0, float[:, :] sound1, bint set_gain_ = False):
    """
    MIX TWO STEREOPHONIC SOUNDS TOGETHER (COMPATIBLE FLOAT32)
    
    * Data samples can have different sizes, the final sound will be the combination of 
    both sounds but the final sound length will be equal to the shortest sound.
    * Both data samples values are multiply by 1/2 before the mix to cap the values 
    
    * Compatible with stereophonic sounds
    
    :param sound0: numpy.ndarray; First sound (stereophonic) 2d numpy array shape (n, 2) float32 datatype
    :param sound1: numpy.ndarray; Second sound to mix (stereophonic). 2d numpy array shape (m, 2) float32
    :param set_gain_: bool; set the gain (increase source data samples values). When set to True the data 
    samples values are multiply par a factor f corresponding to the inverse of the maximum value f=1.0/max_value.
    Both sound data samples will received a proportional volume boost.
    :return : pygame.mixer.Sound; Return a pygame Sound with length equal to the shortest sounds passed as 
    argument to the method.
    """

    if not is_valid_stereo_array(sound0):
        raise ValueError(message26 % 1)

    if not is_valid_stereo_array(sound1):
        raise ValueError(message26 % 2)

    cdef:
        int width0 = <object>sound0.shape[0]
        int width1 = <object>sound1.shape[0]
        int minimum = min(width0, width1)
        int  i = 0
        float [:, :] new_array = empty((minimum, 2), float32)
        float max_v = 0, f = 1.0

    if minimum == 0:
        raise ValueError(message12)

    if set_gain_:
        adaptative_g_stereo_float(sound0)
        adaptative_g_stereo_float(sound1)

    with nogil:

        for i in prange(0, minimum, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_array[i, 0] = <float>(sound0[i, 0] + sound1[i, 0]) / 2.0
            new_array[i, 1] = <float>(sound0[i, 1] + sound1[i, 1]) / 2.0

    return make_sound(asarray(new_array))


cpdef add_mono(sound_array0, sound_array1):
    """
    MIX TWO DATA SAMPLES (ARRAY) TOGETHER  
    
    Both monophonic data samples can be either int16 or float32 data type. 
    Array(s) type int16 will be normalized with the method normalize_array_mono before being added together.
    The algorithm will take the shortest data sample and adjust the sound length to match the other sound using
    LIBROSA time stretching algorithm. time_stretching algorithm takes a lots of cpu resources.
    Prefer sounds with identical length and try to normalized the sounds prior calling this method to 
    get the best performances possible.
          
    * Compatible with monophonic sounds (both data samples must be monophonic type int16 or float32)  
    * Capping values to 1.0, -1.0. 
    To mix two sounds together equally, you just add them together.
    If you know that both sounds will frequently be at or near full amplitude 
    (peak amplitude at or near 1) then you probably want to multiply each sound
    by 0.5 after you add them together, to avoid clipping.
    
    * Returned sound effect will be monophonic or stereophonic 
    
    :param sound_array0: numpy.ndarray; Array shape (n,) monophonic with datatype int16 or float32  
    :param sound_array1: numpy.ndarray; Array shape (n,) monophonic with datatype int16 or float32
    :return: pygame.mixer.Sound. stereophonic if the mixer is setup with 2 channels or monophonic if mixer is set 
    with a single channel
    """

    if not is_valid_mono_array(sound_array0):
        raise ValueError(message11)

    if not is_valid_mono_array(sound_array1):
        raise ValueError(message11)

    cdef:
        int width0 = <object>sound_array0.shape[0]
        int width1 = <object>sound_array1.shape[0]
        int minimum = min(width0, width1)
        int maximum = max(width0, width1)
        #  normalized
        float [::1] s0 = empty(width0, float32)
        float [::1] s1 = empty(width1, float32)
        float [::1] new_array = empty(maximum, float32)
        int i = 0
        float r

    if is_valid_dtype(sound_array0, 'float32'):
        s0 = sound_array0
    else:
        s0 = normalize_array_mono(sound_array0)

    if is_valid_dtype(sound_array1, 'float32'):
        s1 = sound_array1
    else:
        s1 = normalize_array_mono(sound_array1)

    if minimum == 0:
        raise ValueError(message12)

    # stretch shortest sound if size is not equal
    if width0 != width1:
        if minimum == width0:
            s0 = time_stretch(asarray(s0), <float>width0/<float>width1)
        else:
            s1 = time_stretch(asarray(s1), <float>width1/<float>width0)

    with nogil:
        for i in prange(0, maximum, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_array[i] = (s0[i] + s1[i]) * 0.5

    return inverse_normalize_mono(new_array)


cpdef add_stereo(sound0, sound1):
    """
    MIX TWO DATA SAMPLES TOGETHER  
    
    Both stereophonic data samples can be either int16 or float32 data type. 
    Array(s) type int16 will be normalized with the method normalize_array_stereo before being added together.
    The algorithm will take the shortest data sample and adjust the sound length to match the other sound using
    LIBROSA time stretching algorithm. time_stretching algorithm takes a lots of cpu resources.
    Prefer sounds with identical length and try to normalized the sounds prior calling this method to 
    get the best performances possible.
       
    * Compatible with stereophonic sounds (both data samples must be stereophonic type int16 or float32)
    
    * Capping values to 1.0, -1.0. 
    To mix two sounds together equally, you just add them together.
    If you know that both sounds will frequently be at or near full amplitude 
    (peak amplitude at or near 1) then you probably want to multiply each sound
    by 0.5 after you add them together, to avoid clipping.
    
    * Returned sound effect will be monophonic or stereophonic to comply with Pygame mixer settings
    
    :param sound0: numpy.ndarray; Array shape (n, 2) stereophonic with datatype int16 or float32  
    :param sound1: numpy.ndarray; Array shape (n, 2) stereophonic with datatype int16 or float32
    :return: pygame.mixer.Sound. stereophonic if the mixer is setup with 2 channels or monophonic if mixer is set 
    with a single channel
    """


    if not is_valid_stereo_array(sound0):
        raise ValueError(message26 % 1)
    if not is_valid_stereo_array(sound1):
        raise ValueError(message26 % 2)

    cdef:
        int width0 = <object>sound0.shape[0]
        int width1 = <object>sound1.shape[0]
        int minimum = min(width0, width1)
        int maximum = max(width0, width1)
        # decompose stereophonic samples into separate channels and normalized
        float [::1] s0_ch0 = empty(width0, float32)
        float [::1] s0_ch1 = empty(width0, float32)
        float [::1] s1_ch0 = empty(width1, float32)
        float [::1] s1_ch1 = empty(width1, float32)
        float [:, ::1] new_array = empty((maximum, 2), float32)

        int i = 0
        float r0, r1, l0, l1

    if is_valid_dtype(sound0, 'float32'):
        s0_ch0 = sound0[:, 0]
        s0_ch1 = sound0[:, 1]
    else:
        sound0 = normalize_array_stereo(sound0)
        s0_ch0 = ascontiguousarray(sound0[:, 0])
        s0_ch1 = ascontiguousarray(sound0[:, 1])

    if is_valid_dtype(sound1, 'float32'):
        s1_ch0 = sound1[:, 0]
        s1_ch1 = sound1[:, 1]
    else:
        sound1 = normalize_array_stereo(sound1)
        s1_ch0 = ascontiguousarray(sound1[:, 0])
        s1_ch1 = ascontiguousarray(sound1[:, 1])

    if minimum == 0:
        raise ValueError(message12)

    # time_stretch is not compatible with stereophonic data series
    # We have to decompose the stereophonic samples into mono data series
    # and stretch the series separately
    if width0 != width1:
        if minimum == width0:
            l0 = <float>width0/<float>width1
            s0_ch0 = time_stretch(asarray(s0_ch0), l0)
            s0_ch1 = time_stretch(asarray(s0_ch1), l0)
        else:
            l1 = <float>width1/<float>width0
            s1_ch0 = time_stretch(asarray(s1_ch0), l1)
            s1_ch1 = time_stretch(asarray(s1_ch1), l1)

    with nogil:
        for i in prange(0, maximum, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            new_array[i, 0] = (s0_ch0[i] + s1_ch0[i]) * 0.5
            new_array[i, 1] = (s0_ch1[i] + s1_ch1[i]) * 0.5

    return inverse_normalize_stereo(new_array)



cpdef down_sampling_array_stereo(short [:, :] samples_, unsigned short n_=2):
    """
    DOWN SAMPLING STEREOPHONIC ARRAY (INT16) 
        
    * Speed ip sound effect 
    
    To downsample (also called decimate) your signal (it means to reduce the sampling rate), 
    or up-sample (increase the sampling rate) you need to interpolate between your data.
    
    :param samples_: numpy.ndarray; Numpy array shape(n, 2) int16 representing the sampling data (stereophonic)
    :param n_: int; division factor (power of 2 for down-sampling)
    :return: pygame.Sound; return a pygame sound down-sampled 
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef unsigned short int count = 0

    if n_ < 1:
        raise ValueError(message40 % (2, n_))
    elif n_ == 1:
        try:
            return make_sound(samples_)
        except:
            raise ValueError(message39)

    for bit in bin(n_):
        if bit == '1':
            count += 1

    if 0 <= count > 1:
        raise ValueError(message40 % (2, n_))

    cdef:
        int width = <object>samples_[:, 0].shape[0]
        short [:, :] new_array = zeros((width // n_, 2), int16)
        int i, c1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            c1 = i // n_
            new_array[c1, 0] = samples_[i, 0]
            new_array[c1, 1] = samples_[i, 1]

    return make_sound(new_array)


cpdef up_sampling_array_stereo(short [:, :] samples_, unsigned short n_=2):
    """
    UP SAMPLING ARRAY (STEREOPHONIC INT16)
    
    * Slow down sound effect 
    
    :param samples_: numpy.ndarray; Array shape (n, 2) int16 representing the data samples (stereophonic)
    :param n_: int; power of two value for up sampling
    :return: 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef unsigned short int count = 0

    if n_ < 1:
        raise ValueError(message40 % (2, n_))
    elif n_ == 1:
        try:
            return make_sound(samples_)
        except:
            raise ValueError(message39)

    for bit in bin(n_):
        if bit == '1':
            count += 1

    if 0 <= count > 1:
        raise ValueError(message40 % (2, n_))

    cdef:
        int width = <object>samples_.shape[0]
        short [:, :] new_array = zeros((width * n_, 2), int16)
        int i, c1

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width * n_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            c1 = i // n_
            new_array[i, 0] = samples_[c1, 0]
            new_array[i, 1] = samples_[c1, 1]

    return make_sound(new_array)


cpdef slow_down_array_stereo(short [:, :] samples_, int n_):
    """
    CHANGE THE SPEED OF A GIVEN SOUND BY RESAMPLING THE DATA USING LINEAR APPROXIMATION
    
    * Slow down sound effect 
    * Final sound length will be length x n_ (with n>0)
    
    Changing the sound speed is affecting its tempo, pitch and frequency content. 
    When reducing speed, all frequencies become lower. When increasing speed, all frequencies become higher.
    
    :param samples_: ndarray; numpy.ndarray representing a pygame.mixer.Sound object (stereo) 
    :param n_     : int; Resize samples x n_ (if n_ equal 2, the sound length will be twice as much)
    :return       : Return a pygame.mixer.Sound object with a different length 
    """
    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    if n_ < 1:
        raise ValueError(message41 % 2)
    elif n_ == 1:
        try:
            return make_sound(samples_)
        except:
            raise ValueError(message39)

    # LINEAR APPROXIMATION

    cdef:
        int width = <object>samples_.shape[0]
        float [:, ::1] new_array = zeros((width * n_, 2), float32)
        int i = 0, r = 0, ii = 0
        float a, b, inc0, inc1
        float c1 = 1.0 / <float>n_

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(0, (width - 1 - n_) * n_ , n_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            ii = <int>(i * c1)
            a = samples_[ii, 0]
            b = samples_[ii, 1]
            inc0 = (samples_[ii + 1 , 0] - a) * c1
            inc1 = (samples_[ii + 1 , 1] - b) * c1
            for r in range(0, n_):
                new_array[i + r, 0] = a + inc0
                new_array[i + r, 1] = b + inc1

    return make_sound(asarray(new_array, dtype=int16))



cpdef panning_channels_int16(short [::1] channel0_,
                             short [::1] channel1_,
                             short [:, :] samples_,
                             float angle_ = 0.0):
    """
    FOR SOUND PLAYING INDEFINITELY ON THE MIXER (COMPATIBLE INT16)
    
    This method is panning a sound playing on the mixer to a specific angle (argument ange_)
    The sound is panning from -45 degrees to 45 degrees.
    The data samples are modified inplace to reflect the new panning angle.
    Channel0 & Channel1 are used for reference or to keep a fresh copy of the sound channel effect as panning
    a sound result in decreasing the overall volume to zero resulting in loss of channel data 
    when gain volume/reached zero. 
    
    * Compatible with stereo sound object only
    
    :param channel0_: ndarray; take an 1d numpy.ndarray and create a contiguous memoryslice.Raw data represent 
                     the channel0 of the stereo sound   
    :param channel1_: ndarray; take an 1d numpy.ndarray and create a contiguous memoryslice.Raw data represent 
                     the channel1 of the stereo sound
    :param samples_ : ndarray; 2d numpy array representing the sound effect (stereophonic) type int16
    :param angle_   : float; angle in degrees 
    """
    if channel0_ is None:
        raise ValueError(message21 % 1)

    if channel1_ is None:
        raise ValueError(message21 % 2)

    if not is_valid_mono_array(channel0_):
        raise ValueError(message22 % 1)

    if not is_valid_mono_array(channel1_):
        raise ValueError(message22 % 2)

    if -45.0 >= angle_ >= 45.0:
        angle_ = 0

    cdef:
        int width = <object>channel0_.shape[0]
        int i
        float c2 = angle_ * DEG_TO_RADIAN
        float c1 = <float>sqrt(2.0)/2.0
        float volume_left = <float>(c1 * (cos(c2) + sin(c2)))
        float volume_right= <float>(c1 * (cos(c2) - sin(c2)))

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            samples_[i, 0] = <short>(channel0_[i] * volume_left)
            samples_[i, 1] = <short>(channel1_[i] * volume_right)


cpdef panning_channels_float32(float [:] channel0_,
                               float [:] channel1_,
                               float [:, :] samples_,
                               float angle_ = 0.0):
    """
    FOR SOUND PLAYING INDEFINITELY ON THE MIXER (COMPATIBLE FLOAT32)
    
    This method is panning a sound playing on the mixer to a specific angle (argument ange_)
    The sound is panning from -45 degrees to 45 degrees.
    The data samples are modified inplace to reflect the new panning angle.
    Channel0 & Channel1 are used for reference or to keep a fresh copy of the sound channel effect as panning
    a sound result in decreasing the overall volume to zero resulting in loss of channel data 
    when gain volume/reached zero. 
    
    * Compatible with stereo sound object only
    
    :param channel0_: ndarray; take an 1d numpy.ndarray and create a contiguous memoryslice.Raw data represent 
                     the channel0 of the stereo sound   
    :param channel1_: ndarray; take an 1d numpy.ndarray and create a contiguous memoryslice.Raw data represent 
                     the channel1 of the stereo sound
    :param samples_ : ndarray; 2d numpy array representing the sound effect (stereophonic) type float32
    :param angle_   : float; angle in degrees 
    """
    if channel0_ is None:
        raise ValueError(message21 % 1)

    if channel1_ is None:
        raise ValueError(message21 % 2)

    if not is_monophonic(channel0_):
        raise ValueError(message22 % 1)

    if not is_monophonic(channel1_):
        raise ValueError(message22 % 2)

    if -45.0 >= angle_ >= 45.0:
        angle_ = 0

    cdef:
        int width = <object>channel0_.shape[0]
        int i
        float c2 = angle_ * DEG_TO_RADIAN
        float c1 = <float>sqrt(2.0)/2.0
        float volume_left = <float>(c1 * (cos(c2) + sin(c2)))
        float volume_right= <float>(c1 * (cos(c2) - sin(c2)))

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            samples_[i, 0] = <float>(channel0_[i] * volume_left)
            samples_[i, 1] = <float>(channel1_[i] * volume_right)


cpdef panning_sound(sound_, float angle_ = 0.0):

    """
    APPLY A PANNING SOUND EFFECT TO A PYGAME SOUND OBJECT (GIVEN A SPECIFIC ANGLE IN DEGREES) 
    
    This method takes a sound object as argument and will return the equivalent with 
    a panning effect. The original sound raw data is not modified as the method returns 
    a new sound with the panning effect.
    
    * Compatible with stereo sound object only (int16 or float32)
    
    :param sound_ : Sound object: Pygame.Sound object (must be a stereo sound)  
    :param angle_: float; Angle in degrees [-45.0 pass +45.0] -45 pan to the left + 45 pan to the right
    :return      : Sound; Return a pygame.Sound ready to be play on the mixer with the command play()  
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    # Set the panning to zero (no change) but avoid
    # raising an exception
    if -45.0 >= angle_ >= 45.0: angle_ = 0

    try:
        sound_array = sndarray.array(sound_)
    except:
        raise ValueError(message39)

    cdef:
        int width = <object>sound_array.shape[0]
        int channel_number = len(sound_array.shape)

    if channel_number == 1:
        raise ValueError(message4 % sound_array.shape)

    cdef:
        short [:, :] samples_int16 = sound_array if (channel_number == 2 and
                                               sound_array.dtype==int16) else empty((width, 2), int16)
        float [:, :] samples_float32 = sound_array if (channel_number == 2 and
                                               sound_array.dtype==float32) else empty((width, 2), float32)
        int i
        float c2 = angle_ * DEG_TO_RADIAN
        float c1 = <float>sqrt(2.0)/2.0
        float volume_left = <float>(c1 * (cos(c2) + sin(c2)))
        float volume_right= <float>(c1 * (cos(c2) - sin(c2)))

    if width == 0:
        raise ValueError(message12)

    if sound_array.dtype == int16:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                samples_int16[i, 0] = <short>(samples_int16[i, 0] * volume_left)
                samples_int16[i, 1] = <short>(samples_int16[i, 1] * volume_right)
        return make_sound(asarray(samples_int16))

    else:
        with nogil:
            for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                samples_float32[i, 0] = <float>(samples_float32[i, 0] * volume_left)
                samples_float32[i, 1] = <float>(samples_float32[i, 1] * volume_right)
        return make_sound(asarray(samples_float32))




cpdef median_filter_stereo(short [:, :] samples_, unsigned short int dim = 3):
    """
    MEDIAN FILTER A SOUND SAMPLE 
    
    :param samples_: sndarray; numpy.sndarray representing sound samples (stereo only)  
    :param dim     : size of the filter (adjacent values). Default 3, median sort[i-1, i, i+1]  
    :return        : Return a pygame Sound object 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int i, j, l, k
        int width = <object>samples_[:,0].shape[0]
        short [:, :] new_array = zeros((width, 2), int16)
        int m = (dim - 1) >> 1
        int *tmp_ch0 = <int *> malloc(dim * sizeof(int))
        int *tmp_ch1 = <int *> malloc(dim * sizeof(int))
        int *tmp_0 = <int *> malloc(dim * sizeof(int))
        int *tmp_1 = <int *> malloc(dim * sizeof(int))

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            k = 0
            for j in range(-m, m+1):
                l = i - j
                if l < 0: l = 0
                if l > width : l = width
                tmp_ch0[k] = <int>samples_[l, 0]
                tmp_ch1[k] = <int>samples_[l, 1]
                k = k + 1
            tmp_0 = quickSort(tmp_ch0, 0, dim)
            tmp_1 = quickSort(tmp_ch1, 0, dim)
            new_array[i, 0] = <short>tmp_0[m + 1]
            new_array[i, 1] = <short>tmp_1[m + 1]
    return make_sound(asarray(new_array, dtype=int16))



cpdef average_filter_stereo(short [:, :] samples_, unsigned short int dim = 3):
    """
    AVERAGE SOUND SAMPLE VALUES WITH NEIGHBOUR VALUES 
    
    :param samples_: sndarray; Pygame sound samples (stereo)
    :param dim     : int; size of the averaging (neighbours values to include in the calculation) 
    :return        : Pygame Sound object 
    """

    if not is_valid_stereo_array(samples_):
        raise ValueError(message14)

    cdef:
        int i, j, l
        int width = <object>samples_[:,0].shape[0]
        short [:, :] new_array = zeros((width, 2), int16)
        int m = (dim - 1) >> 1
        float c0 = 0.0, c1 = 0.0

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for i in prange(width, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            c0 = 0; c1 = 0
            for j in range(-m, m+1):
                l = i - j
                if l < 0: l = 0
                if l > width - 1 : l = width - 1
                c0 = c0 + samples_[l, 0]
                c1 = c1 + samples_[l, 1]
            c0 = c0 / <float>dim
            c1 = c1 / <float>dim
            new_array[i, 0] = <short>c0
            new_array[i, 1] = <short>c1
    return make_sound(asarray(new_array, dtype=int16))


cpdef echo(sound_, short echoes_, unsigned int sample_rate_, float delay_=1.0):
    """
    CREATE AN ECHO SOUND EFFECT 
    
    * Compatible with monophonic or stereophonic sound object (int16, float32)

    An echo effect causes a sound to repeat on a delay with diminishing volume, 
    simulating the real effect of an echo. 
    
    :param sound_ : pygame.Sound; mono or Stereo sound object, int16 and float32 compatible
    :param echoes_: integer; Number of echo(s) 
    :param sample_rate_    : integer; sample rate (check valid sample rate variable FS)
    :param delay_ : float; time in ms. The amount of time between echoes.
    :return       : return a new sound with an echo sound effect (stereophonic or monophonic)
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    if echoes_ <= 0:
        raise ValueError(message24 % "echoes_")

    if delay_ <= 0:
        raise ValueError(message24 % "delay_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    try:
        array_ = pygame.sndarray.samples(sound_)
    except Exception as e:
        raise ValueError(message39)

    cdef:
        int width = <object>array_.shape[0]
        int channel_number = len(array_.shape)
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)

        short [::1] mono_int16 = zeros(width * echoes_ + c1 * echoes_, int16) if \
            (channel_number==1 and array_.dtype == int16) else empty(width, int16)
        float [::1] mono_float32 = zeros(width * echoes_ + c1 * echoes_, float32) if \
            (channel_number==1 and array_.dtype == float32) else empty(width, float32)
        short [:, :] stereo_int16 = zeros((width * echoes_ + c1 * echoes_, 2), int16) \
            if (channel_number==2 and array_.dtype == int16) else empty((width, 2), int16)
        float [:, :] stereo_float32 = zeros((width * echoes_ + c1 * echoes_, 2), float32) if \
            (channel_number==2 and array_.dtype == float32) else empty((width, 2), float32)

        short [::1] mono_int16_data = array_ if (channel_number==1 and array_.dtype == int16) \
            else empty(width, int16)
        float [::1] mono_float32_data = array_ if (channel_number==1 and array_.dtype == float32) \
            else empty(width, float32)
        short [:, :] stereo_int16_data = array_ if (channel_number==2 and array_.dtype == int16) \
            else empty((width, 2), int16)
        float [:, :] stereo_float32_data = array_ if (channel_number==2 and array_.dtype == float32) \
            else empty((width, 2), float32)


        int i, j, l

    if width == 0:
        raise ValueError(message12)

    if channel_number == 1:
        if array_.dtype == int16:
            with nogil:
                for j in prange(0, echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for i in prange(0, width):
                        l = i + j * width + c1 * j
                        mono_int16[l] = mono_int16_data[i]  / (2 ** j)
            return make_sound(asarray(mono_int16))

        elif array_.dtype == float32:
            with nogil:
                for j in prange(0, echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for i in prange(0, width):
                        l = i + j * width + c1 * j
                        mono_float32[l] = mono_float32_data[i]  / (2 ** j)
            return make_sound(asarray(mono_float32))

    elif channel_number == 2:
        if array_.dtype == int16:
            with nogil:
                for j in prange(0, echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for i in prange(0, width):
                        l = i + j * width + c1 * j
                        stereo_int16[l, 0], stereo_int16[l, 1] = \
                            stereo_int16_data[i, 0]  / (2 ** j), stereo_int16_data[i, 1] / (2 ** j)
            return make_sound(asarray(stereo_int16))

        if array_.dtype == float32:
            with nogil:
                for j in prange(0, echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
                    for i in prange(0, width):
                        l = i + j * width + c1 * j
                        stereo_float32[l, 0], stereo_float32[l, 1] = \
                            stereo_float32_data[i, 0]  / (2 ** j), stereo_float32_data[i, 1] / (2 ** j)
            return make_sound(asarray(stereo_float32))



cpdef echo_mono_float32(sound_, short echoes_, unsigned int sample_rate_, float delay_=1.0):
    """
    CREATE AN ECHO SOUND EFFECT 
    
    * Compatible with monophonic sound object (float32)

    An echo effect causes a sound to repeat on a delay with diminishing volume, 
    simulating the real effect of an echo. 
    
    :param sound_ : pygame.Sound; Monophonic sound object, float32 compatible
    :param echoes_: integer; Number of echo(s) 
    :param sample_rate_    : integer; sample rate (check valid sample rate variable FS)
    :param delay_ : float; time in ms. The amount of time between echoes.
    :return       : return a new sound with an echo sound effect
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    if echoes_ <= 0:
        raise ValueError(message24 % "echoes_")

    if delay_ <= 0:
        raise ValueError(message24 % "delay_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    try:
        array_ = sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if not is_monophonic(array_):
        raise ValueError(message27 % array_.dtype)

    cdef:
        float [:] sound = array_
        int width = <object>sound.shape[0]
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        float [:] new_array = zeros(width * echoes_ + c1 * echoes_, dtype=float32)
        int i, j, l

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in prange(width):
                l = i + j * width + c1 * j
                new_array[l] = sound[i]  / (2 ** j)

    return make_sound(asarray(new_array))

cpdef echo_stereo_float32(sound_, short echoes_, unsigned int sample_rate_, float delay_=1.0):
    """
    CREATE AN ECHO SOUND EFFECT 
    
    * Compatible with stereophonic sound object (float32)

    An echo effect causes a sound to repeat on a delay with diminishing volume, 
    simulating the real effect of an echo. 
    
    :param sound_ : pygame.Sound; Stereo sound object, float32 compatible
    :param echoes_: integer; Number of echo(s) 
    :param sample_rate_    : integer; sample rate (check valid sample rate variable FS)
    :param delay_ : float; time in ms. The amount of time between echoes.
    :return       : return a new sound with an echo sound effect
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    if echoes_ <= 0:
        raise ValueError(message24 % "echoes_")

    if delay_ <= 0:
        raise ValueError(message24 % "delay_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    try:
        array_ = sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if not is_stereophonic(array_):
        raise ValueError(message27 % array_.dtype)

    cdef:
        float [:, :] sound = array_
        int width = <object>sound.shape[0]
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        float [:, ::1] new_array = empty(
            (width * echoes_ + c1 * echoes_, 2), dtype=float32)
        int i, j, l

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in prange(width):
                l = i + j * width + c1 * j
                new_array[l, 0], new_array[l, 1] = sound[i, 0]  / (2 ** j), sound[i, 1] / (2 ** j)

    return make_sound(asarray(new_array))


cpdef echo_mono_int16(sound_, short echoes_, unsigned int sample_rate_, float delay_=1):
    """
    CREATE AN ECHO SOUND EFFECT 

    * Compatible with monophonic sound object (int16)
    
    An echo effect causes a sound to repeat on a delay with diminishing volume, 
    simulating the real effect of an echo. 
    
    * Compatible with stereo sound object only
    
    :param sound_ : pygame.Sound; Monophonic sound int16
    :param echoes_: integer; Number of echo(s)
    :param sample_rate_ : integer; sample rate (check valid sample rate variable FS)
    :param delay_ : float; time in ms. The amount of time between echoes.
    :return       : return a sound with an echo sound effect
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    if echoes_ <= 0:
        raise ValueError(message24 % "echoes_")

    if delay_ < 0:
        raise ValueError(message24 % "delay_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    try:
        array_ = sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if not (is_valid_mono_array(array_) and array_.dtype==int16):
        raise ValueError(message27 % array_.dtype)

    cdef:
        short [:] sound_stereo = array_
        int width = <object>sound_stereo.shape[0]
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        short [:] new_array = zeros(width * echoes_ + c1 * echoes_, dtype=int16)
        int i, j, l

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in prange(width):
                l = i + j * width + c1 * j
                new_array[l] = sound_stereo[i] >> j

    return make_sound(asarray(new_array))

cpdef echo_stereo_int16(sound_, short echoes_, unsigned int sample_rate_, float delay_=1):
    """
    CREATE AN ECHO SOUND EFFECT 

    * Compatible with stereophonic sound object (int16)
    
    An echo effect causes a sound to repeat on a delay with diminishing volume, 
    simulating the real effect of an echo. 
    
    * Compatible with stereo sound object only
    
    :param sound_ : pygame.Sound; Stereo sound int16
    :param echoes_: integer; Number of echo(s)
    :param sample_rate_ : integer; sample rate (check valid sample rate variable FS)
    :param delay_ : float; time in ms. The amount of time between echoes.
    :return       : return a sound with an echo sound effect
    """

    if not is_type_soundobject(sound_):
        raise ValueError(message23 % 1)

    if echoes_ <= 0:
        raise ValueError(message24 % "echoes_")

    if delay_ < 0:
        raise ValueError(message24 % "delay_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    try:
        array_ = sndarray.samples(sound_)
    except:
        raise ValueError(message39)

    if not (is_valid_stereo_array(array_) and array_.dtype==int16):
        raise ValueError(message27 % array_.dtype)

    cdef:
        short [:, :] sound_stereo = array_
        int width = <object>sound_stereo.shape[0]
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        short [:, ::1] new_array = zeros((width * echoes_ + c1 * echoes_, 2), dtype=int16)
        int i, j, l

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in prange(width):
                l = i + j * width + c1 * j
                new_array[l, 0], new_array[l, 1] = sound_stereo[i, 0] >> j, sound_stereo[i, 1] >> j

    return make_sound(asarray(new_array))

# TODO WORKS BUT NEED RE-THINKING
cpdef create_echo_from_channels(short [:] channel0_, short [:] channel1_,
                                short echoes_, int delay_=10, int sample_rate_=44100):
    """
    CREATES A NEW ECHO SOUND EFFECT FROM TWO GIVEN DATA SAMPLES REPRESENTING CHANNEL0 AND CHANNEL1
    
    :param channel0_: numpy.ndarray; Array shape (n, ) with datatype int16 representing the data samples on channel0
    :param channel1_: numpy.ndarray; Array shape (n, ) with datatype int16 representing the data samples on channel1
    :param echoes_  : integer; Number of consecutive echo(s) >= 0
    :param delay_   : integer; Time interval between echo(s) in ms , default 10 milli seconds 
    :param sample_rate_ : float; Sample rate, default 44100 hz
    :return: Return a pygame Sound object with echo(s)
    """

    if not is_valid_mono_array(channel0_):
        raise ValueError()

    if not is_valid_mono_array(channel1_):
        raise ValueError()

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    if not isinstance(echoes_, int):
        raise TypeError(message34 % ("echoes_", "int", type(echoes_)))

    if echoes_ < 0:
        raise ValueError(message35 % ("echoes_", 0, echoes_))

    elif echoes_ == 0:
        return make_sound(asarray(channel0_), dtype=int16)

    if not isinstance(delay_, int):
        raise TypeError(message34 % ("delay_", "int", type(delay_)))

    # if delay_ < 0:
    #     raise ValueError(message35 % ("delay_", 0, delay_))

    cdef:
        int width0 = len(channel0_)
        int width1 = len(channel0_)
        int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        short [:, ::1] new_array  = zeros((width0 * echoes_ + c1 * echoes_, 2), dtype=int16) if c1 > 0 else \
            zeros((width0 * echoes_ - c1 * echoes_, 2), dtype=int16)
        int i, j, l


    if (width0 == 0 or width1 == 0) or (width0 != width1):
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in prange(width0):
                l = i + j * width0 + c1 * j
                new_array[l, 0] = new_array[l, 0] + channel0_[i] >> j
                new_array[l, 1] = new_array[l, 1] + channel1_[i] >> j

    return make_sound(asarray(new_array, dtype=int16))



cpdef create_rev_echo_from_sound(sound_, short echoes_, int delay_=10000, int sample_rate_=44100):
    """
    ECHO EFFECT WITH VOLUME INCREASING 
    
    :param sound_ : pygame.mixer.Sound, Sound can be monophonic or stereophonic  
    :param echoes_: integer; Number of echo(s) >= 0
    :param delay_ : integer; Time interval between echo(s) in samples number (t = samples/sample_rate) >0
    :param sample_rate_: int; Sample rate default is 44100 hz
    :return       : Return a new sound with an inverse echo effect 
    """
    if not isinstance(sound_, SOUNDTYPE):
        raise TypeError(message23 % "sound_")

    if not (sample_rate_ in FS):
        raise ValueError(message15 % (sample_rate_, FS))

    if not isinstance(echoes_, int):
        raise TypeError(message34 % ("echoes_", "int", type(echoes_)))

    if echoes_ < 0:
        raise ValueError(message24 % "echoes_")

    if not isinstance(delay_, int):
        raise TypeError(message34 % ("delay_", "int", type(delay_)))

    if delay_ < 0:
        raise ValueError(message24 % "delay_")

    # Channel separation
    try:
        sound  = sndarray.samples(sound_).astype(dtype=int16, copy=False)
    except:
        raise ValueError(message39)

    cdef:
        short [:] left_ch            = sound[:, 0]
        short [:] right_ch           = sound[:, 1]
        int width                    = left_ch.shape[0]
        # int c1 = <int>(delay_/1000.0 * <float>sample_rate_)
        short [:, ::1] new_array     = zeros((width * echoes_ + delay_ * echoes_, 2), int16)
        int i, j, l, m

    if width == 0:
        raise ValueError(message12)

    with nogil:
        for j in prange(echoes_, schedule=SCHEDULE, num_threads=THREAD_NUMBER):
            for i in range(width - 1, 0, -1):
                l = i + j * width + delay_ * j
                m = echoes_ - j
                new_array[l, 0], new_array[l, 1] = left_ch[i] >> m, right_ch[i] >> m

    return make_sound(asarray(new_array, dtype=int16))


# ----------------------------------------------------------------------------------------- #
#   http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/    #
#   Original concept and coding from zulko (see above link)                                 #
#   Adapted and modify for stereo sounds by yoann Berenguer                                 #
# ------------------------------------------------------------------------------------------#

HANNING = hanning(8192).astype(float32)
DEF ONE_TWELVE = 1.0 / 12.0


cpdef pitchshift(sound_, int n_):
    """
    SOUND PITCH SHIFTING BY N SEMITONES

    * Compatible monophonic and stereophonic sounds

    :param sound_: Sound; Pygame.mixer.Sound object (mono or stereo)
    :param n_    : integer; number of semitones (each n semitones we must multiply the frequency by a factor 2^n/12)
    :return      : return a Pygame sound object with pitch shifted to n semitones
    """
    if not is_type_soundobject(sound_):
        raise TypeError(message23 % "sound_")

    if n_ < 0:
        raise ValueError(message35 % ("n_", 0, n_))

    sound_effect = None

    try:
        array_ = pygame.sndarray.samples(sound_)
    except Exception as e:
        raise ValueError(message39)

    if is_monophonic(array_):
        sound_effect = pygame.sndarray.make_sound(pitchshift_array_mono(array_, n_))

    elif is_stereophonic(array_):
        sound_effect = pygame.sndarray.make_sound(pitchshift_array_stereo(array_, n_))

    else:
        raise ValueError(message30)
    return sound_effect


cdef speedx_mono(numpy.ndarray[float32_t, ndim=1] input_array_, float factor_):
    """
    RETURN DATA SAMPLES WITH PITCH ADJUSTED TO N SEMITONES 
    
    :param input_array_: ndarray; Data samples (1d array type float32) representing the sound samples
    :param factor_     : float; semitones value
    :return            : ndarray; Data samples with pitch adjusted to n semitones
    """

    indices = numpy.round(numpy.arange(0, len(input_array_), factor_))
    indices = indices[indices < len(input_array_)].astype(float32)
    return input_array_[indices.astype(int)]


cdef stretch_mono(float [:] sound_array, float f, int window_size=8192, int h=256):

    cdef:
        numpy.ndarray[float32_t, ndim=1] phase  = zeros(window_size, dtype=float32)
        numpy.ndarray[float32_t, ndim=1] hanning_window = numpy.hanning(window_size).astype(float32)
        numpy.ndarray[complex_t, ndim=1] result = zeros( <int>(len(sound_array) / f) + window_size, complex)
        numpy.ndarray[complex_t, ndim=1] s1 = zeros(window_size, dtype=complex)
        numpy.ndarray[complex_t, ndim=1] s2 = zeros(window_size, dtype=complex)
        numpy.ndarray[complex_t, ndim=1] a2_rephased = zeros(window_size, dtype=complex)
        int i, i2

    for i in numpy.arange(0, len(sound_array)-(window_size+h), h*f):

        # two potentially overlapping subarrays
        a1 = sound_array[<int>i: <int>i + window_size]
        a2 = sound_array[<int>i + h: <int>i + window_size + h]

        # resynchronize the second array on the first
        s1 =  numpy.fft.fft(hanning_window * a1)
        s2 =  numpy.fft.fft(hanning_window * a2)
        phase = (phase + (numpy.angle(s2/s1).astype(float32))) % (2*numpy.pi)
        a2_rephased = numpy.fft.ifft(numpy.abs(s2)*numpy.exp(1j*phase))

        # add to result
        i2 = int(i/f)
        result[<int>i2 : <int>i2 + window_size] += hanning_window * a2_rephased

    result = result/result.max()

    return result.astype(float32)


cpdef pitchshift_array_mono(snd_array_, n_):
    """
    PITCH SHIFTING MONO METHOD 
    
    * This method will works only for mono sound track array (Mono sound converted into a numpy.ndarray)
    
    Pitch-shifting is easy once you have sound stretching. 
    If you want a higher pitch, you first stretch the sound while conserving the pitch,
    then you speed up the result, such that the final sound has the same duration 
    as the initial one, but a higher pitch due to the speed change.

    Doubling the frequency of a sound increases the pitch of one octave, which is 12 musical semitones.
    Therefore to increase the pitch by n semitones we must multiply the frequency by a factor 2^(n/12):
    
    * Input: numpy.ndarray type float32 : representing a mono sound track 
    
    * Output: numpy.ndarray type float32 : New sound with pitch shifted
    
 
    :param snd_array_: ndarray; Data samples to shift (Mono sound array) 
    :param n_        : integer; pitch by n semitones
    :return          : ndarray; Data sample with pitch shifted of n semitones
    """

    cdef:
        float f = (2 ** (<float>n_/12.0))

    stretched = stretch_mono(snd_array_, 1.0/ f)

    # todo this might failed if the data are < 8192
    return speedx_mono(stretched[8192:], f)



cdef speedx_stereo(numpy.ndarray[float32_t, ndim=2] input_array_, float factor_):
    """
    RETURN DATA SAMPLES WITH PITCH ADJUSTED TO N SEMITONES 
    
    :param input_array_: ndarray; Data samples (2d array type float32) representing the sound samples
    :param factor_     : float; semitones value
    :return            : ndarray; Data samples with pitch adjusted to n semitones
    """
    if not is_stereophonic(input_array_):
        raise TypeError(message28)

    if factor_ <= 0:
        raise ValueError(message35)

    # todo investigate but this might be wrong (apply to a single channel)
    cdef numpy.ndarray[float32_t, ndim=1] indices = \
        round(numpy.arange(0, len(<object>input_array_), factor_)).astype(float32)

    return input_array_[indices[indices < len(<object>input_array_)].astype(int), :]


cdef stretch_stereo(numpy.ndarray[float32_t, ndim=2] snd_array_, float factor_):
    """
    STRETCH A STEREO SOUND ARRAY
    
    :param snd_array_: ndarray; Data samples (2d array type float32) representing the sound samples
    :param factor_   : float; semitones value
    :return          : ndarray; Data samples stretch and re-phased (2d array type float32)  
    
    """

    if not is_stereophonic(snd_array_):
        raise TypeError(message28)

    if factor_ <= 0:
        raise ValueError(message35)
    cdef int window_size = 8192
    cdef int h = 512
    cdef:
        int sound_len = <object>snd_array_.shape[0]
        numpy.ndarray[complex_t, ndim=2] result = zeros((<int>(sound_len / factor_) + window_size, 2), complex)
        numpy.ndarray[complex_t, ndim=1] s1 = zeros(window_size, dtype=complex)
        numpy.ndarray[complex_t, ndim=1] s11 = zeros(window_size, dtype=complex)
        numpy.ndarray[complex_t, ndim=1] s2 = zeros(window_size, dtype=complex)
        numpy.ndarray[complex_t, ndim=1] s22 = zeros(window_size, dtype=complex)
        numpy.ndarray[float32_t, ndim=1] phase1 = zeros(window_size, float32)
        numpy.ndarray[float32_t, ndim=1] phase2 = zeros(window_size, float32)
        numpy.ndarray[complex_t, ndim=1] a2_rephased  = zeros(window_size, complex)
        numpy.ndarray[complex_t, ndim=1] a22_rephased  = zeros(window_size, complex)
        numpy.ndarray[float32_t, ndim=1] a1_ch1 = zeros(window_size, float32)
        numpy.ndarray[float32_t, ndim=1] a1_ch2 = zeros(window_size, float32)
        int i, i2

    if sound_len == 0:
        raise ValueError(message12)

    fft_   = fft.fft
    angle_ = numpy.angle
    ffti_  = fft.ifft
    exp_   = numpy.exp
    for i in numpy.arange(0, sound_len - (window_size+h), h * factor_):
        i      = <int>i
        a1_ch1 = snd_array_[i: i + window_size, 0]
        a1_ch2 = snd_array_[i: i + window_size, 1]
        s1     = fft_(HANNING * a1_ch1)
        s11    = fft_(HANNING * a1_ch2)
        a2_ch1 = snd_array_[i + h: i + window_size + h, 0]
        a2_ch2 = snd_array_[i + h: i + window_size + h, 1]
        s2     = fft_(HANNING * a2_ch1)
        s22    = fft_(HANNING * a2_ch2)
        phase1 = (phase1 + angle_(s2 / s1).astype(float32)) % PI2
        phase2 = (phase2 + angle_(s22 / s11).astype(float32)) % PI2
        a2_rephased  = ffti_(numpy.abs(s2) * exp_(1j * phase1))
        a22_rephased = ffti_(numpy.abs(s22) * exp_(1j * phase2))

        i2 = <int>(i / factor_)

        result[i2: i2 + window_size, 0] += HANNING * a2_rephased
        result[i2: i2 + window_size, 1] += HANNING * a22_rephased

    result = result/result.max()

    return result.astype(float32)



cpdef pitchshift_array_stereo(numpy.ndarray[float32_t, ndim=2] snd_array_, int n_):
    """
    PITCH SHIFTING STEREO METHOD 
    
    * This method will works only for stereo sound track arrays (Stereo sound converted into a numpy.ndarray)
    
    Pitch-shifting is easy once you have sound stretching. 
    If you want a higher pitch, you first stretch the sound while conserving the pitch,
    then you speed up the result, such that the final sound has the same duration 
    as the initial one, but a higher pitch due to the speed change.

    Doubling the frequency of a sound increases the pitch of one octave, which is 12 musical semitones.
    Therefore to increase the pitch by n semitones we must multiply the frequency by a factor 2^(n/12):
    
    * Input: numpy.ndarray type float32 : representing a stereo sound track 
    
    * Output: numpy.ndarray type float32 : New sound with pitch shifted
    
 
    :param snd_array_: ndarray; Data samples to shift (stereo sound array) 
    :param n_        : integer; pitch by n semitones
    :return          : ndarray; Data sample with pitch shifted of n semitones
    """

    if not is_stereophonic(snd_array_):
        raise TypeError(message28)

    cdef:
        float f = (2 ** (n_ * ONE_TWELVE))
        numpy.ndarray[float32_t, ndim=2] stretched = stretch_stereo(snd_array_, 1.0 / f)
    # todo this might failed if the data are < 8192
    return speedx_stereo(stretched[8192:], f)


# ----------------------------------------------------------------------------------------- #
#   Please visit the following pages for more information concerning methods defined below  #
#   https://github.com/librosa/librosa                                                      #
#   https://librosa.org/doc/latest/index.html                                               #
#   LIBROSA                                                                                 #
# ------------------------------------------------------------------------------------------#


# ****************  BELOW METHODS ARE CONVENIENT HOOKS TO LIBROSA METHODS *******************
cpdef shift_pitch(y_, float sr_, int steps_):
    """
    HOOK TO LIBRARY LIBROSA SHIFT_PITCH 
    
    Shift the pitch of a waveform by n_steps steps.
    A step is equal to a semitone if bins_per_octave is set to 12.
    
    :param y_    : np.ndarray [shape=(n,)] audio time series
    :param sr_   : float number > 0 [scalar] audio sampling rate of y
    :param steps_: integer [scalar] how many (fractional) steps to shift y
    :return      : numpy.ndarray; The pitch-shifted audio time-series
    """
    # todo select channels
    return librosa.effects.pitch_shift(y_, sr_, n_steps=steps_)


cpdef time_stretch(y_, rate_):
    """
    HOOK TO LIBRARY LIBROSA TIME_STRETCH
    
    Time-stretch an audio series by a fixed rate.
    
    :param y_   : np.ndarray [shape=(n,)]; audio time series
    :param rate_: float > 0 [scalar] Stretch factor. If rate > 1, 
    then the signal is sped up. If rate < 1, then the signal is slowed down.
    :return: np.ndarray [shape=(round(n/rate),)]; audio time series stretched by the specified rate
    """
    return librosa.effects.time_stretch(y_, rate_)



# ----------------------------------------------------------------------------------------- #
#     * COPYRIGHT 1999-2015 Stephan M. Bernsee <s.bernsee [AT] zynaptiq [DOT] com>
#     *
#     * 						The Wide Open License (WOL)
#     *
#     * Permission to use, copy, modify, distribute and sell this software and its
#     * documentation for any purpose is hereby granted without fee, provided that
#     * the above copyright notice and this license appear in all source copies.
#     * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
#     * ANY KIND. See http://www.dspguru.com/wol.htm for more information.
# ------------------------------------------------------------------------------------------#

cpdef PitchShiftingBernsee(float pitchShift, int numSampsToProcess, int fftFrameSize, int osamp,
         int sampleRate, float [:] indata, float [:] outdata):
    """
    PITCH SHIFTING (ALGORITHM FROM Stephan M. Bernsee) 
    
    * COPYRIGHT 1999-2015 Stephan M. Bernsee <s.bernsee [AT] zynaptiq [DOT] com>
    *
    * 						The Wide Open License (WOL)
    *
    * Permission to use, copy, modify, distribute and sell this software and its
    * documentation for any purpose is hereby granted without fee, provided that
    * the above copyright notice and this license appear in all source copies.
    * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
    * ANY KIND. See http://www.dspguru.com/wol.htm for more information.
    
    :param pitchShift: float; Pitch shift factor (float) [0.5 (one octave down) ... 2.(one octave up).]. 
    A value of 1 does not change the pitch
    :param numSampsToProcess: int; tells the routine how many samples in indata[0...numSampsToProcess-1] should be
    pitch shifted and moved to outdata[0 ... numSampsToProcess-1].The two buffers can be identical,
    (ie. it can process the data in-place). 
    :param fftFrameSize: int; defines the FFT frame size used for the processing. Typical values are 1024, 2048 and 4096. 
    It may be any value <= MAX_FRAME_LENGTH but it MUST be a power of 2.
    :param osamp: int; osamp is the STFT oversampling factor which also determines the overlap between adjacent STFT
    frames. It should at least be 4 for moderate scaling ratios. A value of 32 is recommended for best quality.
    :param sampleRate: int; takes the sample rate for the signal in unit Hz, ie. 44100 for 44.1 kHz audio.
    :param indata: numpy.ndarray; Numpy ndarray data buffer shape (n, ) with datatype float 32. The data passed 
    to the routine in indata[] should be in the range [-1.0, 1.0), which is also the output range for the data, 
    make sure you scale the data accordingly (for 16bit signed integers you would have to divide 
    (and multiply) by 32768).
    :param outdata: numpy.ndarray; Numpy ndarray data buffer shape (n, ) with datatype float 32. Empty buffer to pass 
    to the routine (The buffer will be filled inplace).
    :return: void
    """
    smbPitchShift(pitchShift, numSampsToProcess, fftFrameSize, osamp, sampleRate, &indata[0], &outdata[0])


