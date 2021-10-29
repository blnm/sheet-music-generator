# librares
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

# our files
import NoteDetector
from utils import f2
import utils

def pick_best_frequencies(input_freqs, input_ints):
    # recursively get all frequencies above (or only a little below) the mean intensity
    # note that there shouldn't reeeeeally be notes that much more quiet than the mean intensity
    # unless they're using the sustain pedal but I'm not sure we're resilient enough to noise 
    # to be confident enough to attempt to detect that. Maybe this changes in the future, idk

    inds = np.where(input_ints > np.mean(input_ints) * 0.8) # tried .7 but had some false positives still
    output_freqs = input_freqs[inds]
    output_ints = input_ints[inds]

    if (len(input_freqs) == len(output_freqs)):
        return output_freqs, output_ints

    elif (len(output_freqs) < len(input_freqs)):
        return pick_best_frequencies(output_freqs, output_ints)
    else:
        #??????????????????????????????????????????????????????????????????????????????????????
        print("woah there")
        return input_freqs, input_ints


def get_frequencies(audio_seg, samplerate):
    N = len(audio_seg)  # number of total samples (that we're currently testing
    T = 1/samplerate

    x = np.linspace(0.0, N*T, N, endpoint=False)    # x axis - initialize to 0s
    y = np.array(audio_seg)                         # y axis - initialize to raw audio - no specific reason for this...? this is from their docs but pretty sure we can use audio_seg directly.
    yf = fft(y)                                     # fft on y axis - this gives list of intensities
    xf = fftfreq(N, T)[:N//2]                       # get frequencies on x axis - this gets the frequencies that correspond to the intensities in yf

    yfr = 2.0/N * np.abs(yf[0:N//2])                # convert yf to real numbers and multiply by 2N per DFT - 2N not really necessary as output is somewhat subjective but may as well.
    xfr = xf[0:len(yfr)]

    #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])) # opens a plot window and pauses until closed
    #plt.grid()         # actually super helpful for debugging
    #plt.show()         # because there's usually a SHIT TON of detected frequencies (i.e. 49.998, 49.999, 50, 50.001, etc.)

    best_freqs, best_ints = pick_best_frequencies(xfr, yfr)
    
    #plt.plot(xfr, yfr) # opens a plot window and pauses until closed
    #plt.grid()         # actually super helpful for debugging
    #plt.show()         # because there's usually a SHIT TON of detected frequencies (i.e. 49.998, 49.999, 50, 50.001, etc.)

    return best_freqs, best_ints

def get_note_array(audio_seg, samplerate):
    freqs, ints = get_frequencies(audio_seg, samplerate)

    #if(len(freqs) > 5):
    #    freqs = freqs

    frame = NoteDetector.Frame()
    for freq,int in zip(freqs, ints):
        if freq != 0.0:
            frame.Add(NoteDetector.Note(freq, int))

    return frame


#wavfile = read("C:\\Users\\Mitchell\\Desktop\\capstone_test_wavs\\PianoTest1_mono.wav")
wavfile = read("C:\\Users\\Mitchell\\Desktop\\capstone_test_wavs\\tone_440-220.wav")
audio = wavfile[1]
samplerate = wavfile[0] # standard is 44.1kHz but in yiruma sample its 48kHz for some reason

# need to find time per sample, create rolling window w/ a reasonable sample size per window


# NOTE this is not at all a reasonable number I'm just using it for testing
# the "window resolution" is the chunks of time that we want to detect frequencies in
# the larger the number, the more accurate the frequencies
# the smaller the number, the more accurate the time
# larger numbers are also SUBSTANTIALLY faster. As there isn't a huge difference between how long it takes to process a frame, regardless of how large it is.
# I *think* we should aim for around 100ms, but maybe smaller. We have to decide how fast music we want to decypher.
# additionally, the lower the resolution, the faster we can theoretically output data. If we're really commited to zero-delay real-time output (which we're not),
# having this number smaller lowers the delay between output. Theoretically, there's no reason to delay output beyond this window... so real-time isn't inherently out-of-reach
window_resolution = 100.0 #ms

sample_count_per_window = window_resolution * (samplerate / 1000.0) # convert cyc/sec to cyc/msec, multiply by the number of msec we want = number of cycles (frames of original audio) that we want.

number_windows = int(len(audio) / sample_count_per_window)
print("number of windows: ", number_windows)

# NOTE
# window sizes have a minumum
# so if we get a fractional window we just make a bigger window on the last one
# not really an issue for the get_max_frequency() function because it dynamically determines sizes and such
# plus window sizes vary on WAVE sample rate so hard coded shit would be bad
# we casted to int above so have to set window size specifically in loop below

for i in range(0, number_windows):
    upperbound = int((i+1) * sample_count_per_window)
    if (i == number_windows-1):
        upperbound = int(len(audio))
    
    notes = get_note_array(audio[i : upperbound], samplerate)

    print(str(f2(100 * i / number_windows)) + "%", notes)

print("------")
