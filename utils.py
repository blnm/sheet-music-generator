
FREQUENCY_ERROR = 1.0   # unsure of what to do for this... Lowest note on piano is A0, 2nd is A#0, which have freqs of 27.5 and 29.14 respectfully
                        # but at a higher frequency small errors become a much larger frequency offset. Should probably use a multiplier (i.e. 5-7%).
                        # if we do any more than ~1.5Hz we can have a user play A0 and A#0 simultaneously and not be sure it isn't a FFT error.

highest_freq = 4186     # highest
rounded_hi_freq = 4500  # rounded for breathing room - can lower this later for speed

def f2(f):
    return "{:.2f}".format(f)
