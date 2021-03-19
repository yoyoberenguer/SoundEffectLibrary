# Python SoundEffectLibrary (Cython file)

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

List of methods available for version 1.0.1
- Microphone recording 
- Sound recording (wav format)
- Data sample normalisation &reverse normalisation process 
- RMS calculator and display
- Fade in / Fade out effect
- Tinnitus effect
- Generate silence 
- Low pass filter 
- High pass filter
- Band pass filter
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
```python
# In a command prompt and under the directory containing the source files
C:\>python setup_project.py build_ext --inplace
...
...
...
   Creating library build\temp.win-amd64-3.6\Release\SoundEffectLib.cp36-win_amd64.lib and object build\temp.win-amd64-3.6\Release\SoundEffectLib.cp36-win_amd64.exp
Generating code
Finished generating code


# If the compilation fail, refers to the requirement section and make sure cython 
# and a C-compiler are correctly install on your system. 
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
It it sometimes required to normalised data sample before processing when the mixer is 
initialised in 16bit (signed or unsigned int mode).
The below methods will convert monophonic and stereophonic 16bit audio sample into an 
32bit equivalent format. The normalisation is a straight forward calculation, as a result it can be 
used in real time processing without degrading performances of your game / application.
In addition you can use the reverse normalisation process to do the opposite when required
(32bit data sample converted into 16bit signed data sample).
```
```python
# usage for a monophonic audio sample signed 16bit format
normalize_array_mono(array)

# usage for a stereophonic audio sample signed 16bit format
normalize_array_stereo(array)

# usage for a sound object signed 16bit (monophonic or stereophonic modes) 
normalize_sound(sound)

# usage for inverse normalisation monophonic audio sample 32bit format
inverse_normalize_mono(array)

# usage for inverse normalisation stereophonic audio sample 32bit format
inverse_normalize_stereo(array)

# usage for inverse normalisation stereophonic audio sample 32bit format
inverse_normalize_stereo_asarray(array)

# usage for inverse normalisation monophonic audio sample 32bit format
inverse_normalize_mono_asarray(array)
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/normalisation.png)

### RMS calculation
```
An analysis used for the overall amplitude of a signal is called the root-mean-square (RMS)
amplitude or level. Conceptually, it describes the average signal amplitude. However, it is
different than simply measuring the arithmetic mean of a signal.An audio signal can have 
both positive and negative amplitude values. If we took the arithmetic mean of a sine wave,
the negative values would offset the positive values and the result would be zero. This 
approach is not informative about the average signal level.
This is where the RMS level can be useful. It is based on the magnitude of a signal as a 
measure of signal strength, regardless of whether the amplitude is positive or negative.
The magnitude is calculated by squaring each sample value (so they are all positive), then 
the signal average is calculated, eventually followed by the square root operation. More 
completely, the RMS level is, “the square root of the arithmetic mean of the signal squared.”
https://www.hackaudio.com/digital-signal-processing/amplitude/rms-amplitude/

Return the value(s) in decibels (single value for a monophonic array and 3 values for
stereophonic array such as (left channel, right channel, stereo)
```
```python
# Return the RMS value (in decibels), sound_object is a pygame Sound object such 
# as sound_object = mixer.Sound(os.path.join(my_directory, '', 'sound_name.ogg'))
rms_value_mono(sound_object)

# Return 3 values, left channel, right and centre (all values in decibels)
rms_values_stereo(sound_object)

# Display the rms value(s) (classic print)
show_rms_values(sound_object)

# Function returning MIN, AVG and MAX values of a sound object (scalar values, not
# rms value(s))
display_sound_values(sound_object)
```
```python
```

### Fade in & Fade out effect 
```
In audio engineering, a fade is a gradual increase or decrease in the level of an audio 
signal.A recorded song may be gradually reduced to silence at its end (fade-out), or may 
gradually increase from silence at the beginning (fade-in). 
```
```python
# Fade in effect by passing a sound object, fade times (in seconds) and the audio sample rate
fade_in(sound_, fade_in_, sample_rate_)

# See also designated methods for monophonic or stereophonic data samples (int16 or float32 bit format)
fade_in_mono_int16()
fade_in_mono_float32()
fade_in_stereo_int16()
fade_in_stereo_float32()
fade_in_mono_inplace_int16()
fade_in_mono_inplace_float32()
fade_in_stereo_inplace_int16()
fade_in_stereo_inplace_float32()
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/FadeInEffect.png)

```python
# Fade out sound object (sound_) given a starting time (fade_out_ in seconds) and the sample rate 
fade_out(sound_, fade_out_, sample_rate_)

# See also dedicated methods for monophonic & stereophonic data samples (int16 & float32)
fade_out_mono_int16()
fade_out_mono_float32()
fade_out_stereo_int16()
fade_out_stereo_float32()
fade_out_mono_inplace_int16()
fade_out_mono_inplace_float32()
fade_out_stereo_inplace_float32()
fade_out_stereo_inplace_int16()
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/FadeOutEffect.png)

### Tinnitus effects
```
Create the tinnitus effect (unpleasant ringing or buzzing in the ears)
This effect can be use in your video game after a loud explosion 
You can choose a specific frequency, duration and amplitude for the effect. 
tinnitus_fade_out method include a fade-out effect for a smooth transition to the 
next sound effect.
The sound object can be monophonic or stereophonic (datatype int16 or float32)
```
```python
sound = tinnitus_fade_out(0.5, duration_=5.0, frequency_=5000)
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/TinnitusFadeOut.png)

### Generate a silence in the data sample  
```
Create a silence in your data samples (signal length remain unchanged).
This function cancel any sound effect in between a specific interval of your data 
sample.
```
```python
# create a silence (serie of zeros data) in the interval start_ & end_ (time in seconds) 
sound = generate_silence(sound_, start_, end_, sample_rate_)
```
```python
# See also dedicated methods for monophonic & stereophonic data samples (int16 & float32)
# monophonic array, int16 
generate_silence_array_mono_int16()

# monophonic array float32
generate_silence_array_mono_float32()

# stereophonic array int16
generate_silence_array_stereo_int16()

# stereophonic arrat float32
generate_silence_array_stereo_float32()
```

### Low pass filtering effect (signal processing)
```
Low-pass filters pass through frequencies below their cutoff frequencies, and progressively
attenuates frequencies above the cutoff frequency. Low-pass filters are used in audio 
crossovers to remove high-frequency content from signals being sent to a low-frequency subwoofer
system.
```
```python
# apply a low pass filter to the sound effect (sound_) with the cut frequency defined by the 
# variable fc_
sound = low_pass(sound_, float fc_):
```
```python
# See also dedicated methods for monophonic & stereophonic data samples (int16 & float32)
# Below low pass filter will be applied inplace 

# low pass filter on a monophonic array int16
low_pass_mono_inplace_int16(sound_array_, fc_)

# low pass filter on a monophonic array float32
low_pass_mono_inplace_float32(sound_array_, fc_)

# low pass filter on a stereophonic array int16
low_pass_stereo_inplace_int16(sound_array_, fc_)

# low pass filter on a stereophonic array float32
low_pass_stereo_inplace_float32(sound_array_, fc_)
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/LowPassFilter.png)

### High pass filtering effect (signal processing)
```
high-pass filters passes high frequencies fairly well; it is helpful as a filter to cut any unwanted 
low-frequency components.
```

```python
# apply a high pass filter to the sound effect (sound_) with the cut frequency defined by the 
# variable fc_
sound = high_pass(sound_, fc_)
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/HighPassFilter.png)

### Bandpass filtering effect (signal processing)
```
high-pass filters passes a limited range of frequencies.
```
```python
# apply a bandpass filter to the sound effect (sound_) with the cut frequency defined by the 
# variable fc_
sound =  band_pass(sound_, fc_low, fc_high, order=5)
```

![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/passbandFilter.png)

### Harmonic (display signal frequency domain)
```
A harmonic of such a wave is a wave with a frequency that is a positive integer multiple of the frequency
of the original wave, known as the fundamental frequency. The original wave is also called the 1st harmonic,
the following harmonics are known as higher harmonics. As all harmonics are periodic at the fundamental 
frequency, the sum of harmonics is also periodic at that frequency. For example, if the fundamental frequency
is 50 Hz, a common AC power supply frequency, the frequencies of the first three higher harmonics are 100 Hz 
(2nd harmonic), 150 Hz (3rd harmonic), 200 Hz (4th harmonic) and any addition of waves with these frequencies 
is periodic at 50 Hz.

Harmonics method will return 2 objects the first one is a pygame surface showing the frequency analysis (signal 
amplitude versus frequency), note that the signal amplitude is not in decibels. 
The second object is the data containing all the frequency values (numpy.ndarray shape (n, ) float32). 
You can specify the size of the pygame surface (24bit) with the variable width and height (default is 255x255 
pixels). The sample data must be a numpy.ndarray shape (n, ) or (n, 2) int16 or float32. The spectrum analysis 
will be applied to the first channel of a stereophonic sound. 
```
```python
# Return a pygame surface size 255x255 pixels and array containing all frequencies
# To display the surface, blit it to the video mode using the pygame method blit 
# You can also save the surface to disk using the command pygame.image.save(surface_, "myplot.png")
surface_, array_ = harmonics(samples_, sampling_rate_=44100, width=255, height=255)

```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/Harmonics.png)
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/harmonics_details.png)

### Remove silence from data sample
```
Trim leading and trailing silence from an audio signal 
This method detect in the audio singal values that are below a specific rms threshold (rms_threshold_) 
and disregard these values when building the output audio signal. The comparison is applied only at
the start and at the end of the record to avoid deleting short silence in an audio signal. 
A new array is build to match the original audio signal minus the deleted records.
```
```python
# Init the pygame mixer is stereo and signed int16 values
SAMPLE_RATE = 44100
pygame.mixer.init(SAMPLE_RATE, -16, 2, 2048, allowedchanges=0)
# load a sound effect into a numpy.array shape(n, 2) stereophonic sound
array_ = pygame.sndarray.samples(sound)
# sound length calculation 
sound_length = array_.shape[0] / SAMPLE_RATE
# Determine the average RMS value for the entire array 
rms = rms_values_stereo(array_)[2]
# remove silence from the data samples (values below -35.1 decibels will be ignored during the 
# contruction of the new sound object)
new_sound = remove_silence_stereo_int16(array_, -35.1)        
 
```
```python
# See also dedicated methods for monophonic & stereophonic data samples (int16 & float32)
remove_silence_stereo_int16(short [:, :] samples_, rms_threshold_=None, bint bypass_avg_=False)
remove_silence_stereo_float32(float [:, :] samples_, rms_threshold_=None, bint bypass_avg_=False)
remove_silence_mono_int16(short [::1] samples_, rms_threshold_=None, bint bypass_avg_ = False)
remove_silence_mono_float32(float [::1] samples_, rms_threshold_=None, bint bypass_avg_ = False)
```

### Create basic audio signals (noise, square, triangular, cosinus, waveform carrier)
```
Create/design periodic signals such as square, cosinus or triangular using a set of variables to 
define the signal properties (frequency, amplitude, duration, phase etc). 
Finally, a sound object is build from the data samples and can be play by the mixer.
These functions can be useful for generating basic sound for a specific frequency or for mixing
a variety of signals together for testig purpose, signal processing or electronics project etc.
The noise signal is a random generated data samples, this function as no other purpose than generating 
a random noise to a loudspeakers.
The waveform carrier is a tool that can build a sound effect with a serie of specific frequencies 
passed as argument to the function (all frequencies will be mixed together). Each sub frequency 
singnal will have their amplitude decreased compare to first frequency (carrier signal).

```
```python
# Create a noise signal (random generated data), 1 second and ampliude max
noise_signal(amplitude_ = 1.0, duration_  = 0.5, sample_rate_ = 44100)
# Create a square generated signal, amplitude max, 1 second, freq 100 hz.
square_signal(amplitude_ = 1.0, duration_  = 1.0, frequency_ = 100, sample_rate_ = 48000,
                    c_ = 0.0, phi_  = 0.0)
# Create a triangular signal of 100hz                     
triangular_signal(amplitude_ = 1.0, duration_  = 0.5, frequency_ = 100, sample_rate_ = 44100,
                        ramp_ = 0.5)
# Create a cos signal, max amplitude, 100 hz                
cos_signal(amplitude_ = 1.0, duration_ = 1.0, frequency_ = 100, sample_rate_ = 48000,
           c_ = 0.0, phi_ = 0.0)
# Create a carrier with modulated frequencies (100, 200, 300)
cos_carrier(amplitude_ = 1.0, duration_  = 1.0, frequencies_ = [100, 200, 300], sample_rate_ = 44100,
                  c_ = 0.0,  phi_ = 0.0)
```
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/NoiseSignal.png)
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/SquareSignal.png)
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/TriangularSignal.png)
![alt text](https://github.com/yoyoberenguer/SoundEffectLibrary/blob/main/screenshots/CosineSignal.png)

### Time shifting 
```
Time shifting allow a signal to be shifted in a specific time window (defined by the signal length)
The total amount of time shifting cannot exceed the signal length. Values shifted outside 
the window are deleted and will not be included in the final sound object
```
```python
# See dedicated methods for monophonic & stereophonic data samples (int16 & float32)

# Array monophonic int16 and float32
time_shift_mono_int16(samples_, shift_, sample_rate_)
time_shift_mono_float32(samples_, shift_, sample_rate_)

# Array stereophonic int16 and float32
time_shift_stereo_int16(samples_, shift_, sample_rate_)
time_shift_stereo_float32(samples_, shift_, sample_rate_)

# Shift a specific channel (samples_ must be a stereophonic array shape (n, 2))
time_shift_channel(samples_, shift_, sample_rate_, int channel_=0)
```


### Volume control  
```
This will set the playback volume (loudness) for this Sound. 
This will immediately affect the Sound if it is playing. It will also affect any future playback of this Sound.
Volume change is applied inplace. Maximum volume is 1.0 
```
```python
set_volume(sound_, float volume_=1.0)

```

### Reversing sound effect (playing backward)
```
Reverse the selected audio, so that the end of the audio will be heard first and the beginning last.
```
```python
reverse_sound(sound_)
```
```python
# Reverse the sound (compatible mono and stereo int16 or float32)
reverse_sound_beta(sound_)

# monophonic int16 and float32
reverse_stereo_int16(samples_)
reverse_stereo_float32(samples_)

# stereophonic int16 and float32
reverse_mono_int16(samples_)
reverse_mono_float32(samples_)
```

### Inversion sound effect 
```
Invert flips an audio samples upside-down, reversing their polarity. 
The positive samples are moved below the zero line (so becoming negative), and negative samples 
are made positive. Invert does not usually affect the sound of the audio at all, but it can be used 
for audio cancellation. 
```
```python

# monophonic int16 and float32
invert_array_mono_int16(samples_)
invert_array_mono_float32(samples_)

# stereophonic int16 and float32
invert_array_stereo_int16(samples_)
invert_array_stereo_float32(samples_)
```
```python
```

### Mixing sounds together 
```
Adding / mixing sounds together to create a more complex sound effect. 
You can design you own sound effect by mixing two sound arrays (same length) together 
to create a third sound and save it onto disk.
```
```python
# Add two sounds together type int16 stereophonic, set_gain is set to False by default
# set_gain True, raise the gain/volume for both sounds by a factor f=1.0 / max_value
adding_stereo_int16(sound0, sound1, set_gain_ = False)
# Add two sounds together stereophonic type float32 
adding_stereo_float32(sound0, sound1, set_gain_ = False)
add_mono(sound_array0, sound_array1)
add_stereo(sound0, sound1)
```


### Downsampling upsampling audio signals
```
```
```python
down_sampling_array_stereo(samples_, n_=2)
up_sampling_array_stereo(samples_, n_=2)
slow_down_array_stereo(samples_, n_)
```
```python
```

### Panning sound effect
```
<<WIKIPEDIA>>
Panning is the distribution of a sound signal (either monaural or stereophonic pairs) 
into a new stereo or multi-channel sound field determined by a pan control setting.
A typical physical recording console has a pan control for each incoming source channel. 
A pan control or pan pot (short for "panning potentiometer") is an analog control with 
a position indicator which can range continuously from the 7 o'clock when fully left 
to the 5 o'clock position fully right. Audio mixing software replaces pan pots with 
on-screen virtual knobs or sliders which function like their physical counterparts.
```
```python
# his method is panning a sound playing on the mixer to a specific angle (argument ange_) 
# The data samples are modified **INPLACE** to reflect the new panning angle.
panning_channels_int16(channel0_, channel1_, samples_, angle_ = 45.0)
panning_channels_float32(channel0_, channel1_, samples_, angle_ = -45.0)

# This method takes a sound object as argument and will return the equivalent with 
# a panning effect. The original sound raw data is not modified as the method returns 
# a new sound with the panning effect.
panning_sound(sound_, angle_ = 0.0)
```
```python
```

### Median and averaging filter (signal processing)
```
The median filter provides a means for dealing with "spiky" noise and separating peaks 
from a slowly changing baseline, even when the exact nature of the drift and noise distribution
is not known. Median filtering is a useful and complementary addition to existing digital 
filtering techniques
```
```python
# Apply a median filtering on data samples shape (n, 2) int16 ** not compatible with float32
median_filter_stereo(samples_, dim = 3)

# Apply neighbourhood averaging on data shape (n, 2) int16 ** not compatible with float32
average_filter_stereo(samples_, dim = 3)
```

### Create echo in the data sample
```
<<WIKIPEDIA>>
In audio signal processing and acoustics, echo is a reflection of sound that arrives at the
listener with a delay after the direct sound. The delay is directly proportional to the distance 
of the reflecting surface from the source and the listener. Typical examples are the echo produced 
by the bottom of a well, by a building, or by the walls of an enclosed room and an empty room.
A true echo is a single reflection of the sound source.
```
```python
echo(sound_, echoes_, sample_rate_, delay_=1)
echo_mono_float32(sound_, echoes_, sample_rate_, delay_=1.0)
echo_stereo_float32(sound_, echoes_, sample_rate_, delay_=1.0)
echo_mono_int16(sound_, echoes_, sample_rate_, delay_=1):
echo_stereo_int16(sound_, echoes_, sample_rate_, delay_=1):
create_echo_from_channels(channel0_, channel1_, echoes_, delay_=10, sample_rate_=44100)
create_rev_echo_from_sound(sound_, echoes_, delay_=10000, sample_rate_=44100)
```


### Time stretching and pitch shifting
```
Pitch shifting is a sound recording technique in which the original pitch of a sound is raised or 
lowered. Effects units that raise or lower pitch by a pre-designated musical interval are called
pitch shifters.
```
```python
# Please visit : 
# http://zulko.github.io/blog/2014/03/29/soundstretching-and-pitch-shifting-in-python/    #
#   Original concept and coding from zulko (monophonic version)                           #
pitchshift(sound_, n_)


# ----------------------------------------------------------------------------------------- #
#   Please visit the following pages for more information concerning methods defined below  #
#   https://github.com/librosa/librosa                                                      #
#   https://librosa.org/doc/latest/index.html                                               #
#   LIBROSA                                                                                 #
# ------------------------------------------------------------------------------------------#
shift_pitch(y_, sr_, steps_)

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

PitchShiftingBernsee(pitchShift, numSampsToProcess, fftFrameSize, osamp, sampleRate, indata, outdata)
```
```python
```
