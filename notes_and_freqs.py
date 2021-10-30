
import utils

def get_closest_note_for_freq(frequency: float):
    curr_dist = 9999 # bigger than all freqs
    curr_note = "C0"
    for note_freq in freqs:
        if abs(note_freq - frequency) <= curr_dist:
            curr_dist = abs(note_freq - frequency)
            curr_note = notes[freqs.index(note_freq)]
        else: # distance is increasing, we've got our best match
            return curr_note
    return curr_note

def get_distance_from_note(frequency: float, note: str):
    return abs(frequency - freqs[notes.index(note)])

notes = [   "C0",   "Db0",  "D0",   "Eb0",  "E0",   "F0",   "Gb0",  "G0",   "Ab0",  "A0",   "Bb0",  "B0",
            "C1",   "Db1",  "D1",   "Eb1",  "E1",   "F1",   "Gb1",  "G1",   "Ab1",  "A1",   "Bb1",  "B1",
            "C2",   "Db2",  "D2",   "Eb2",  "E2",   "F2",   "Gb2",  "G2",   "Ab2",  "A2",   "Bb2",  "B2",
            "C3",   "Db3",  "D3",   "Eb3",  "E3",   "F3",   "Gb3",  "G3",   "Ab3",  "A3",   "Bb3",  "B3",
            "C4",   "Db4",  "D4",   "Eb4",  "E4",   "F4",   "Gb4",  "G4",   "Ab4",  "A4",   "Bb4",  "B4",
            "C5",   "Db5",  "D5",   "Eb5",  "E5",   "F5",   "Gb5",  "G5",   "Ab5",  "A5",   "Bb5",  "B5",
            "C6",   "Db6",  "D6",   "Eb6",  "E6",   "F6",   "Gb6",  "G6",   "Ab6",  "A6",   "Bb6",  "B6",
            "C7",   "Db7",  "D7",   "Eb7",  "E7",   "F7",   "Gb7",  "G7",   "Ab7",  "A7",   "Bb7",  "B7",
            "C8",   "Db8",  "D8",   "Eb8",  "E8",   "F8",   "Gb8",  "G8",   "Ab8",  "A8",   "Bb8",  "B8"
        ]

freqs = [   16.35,      17.32,      18.35,      19.45,      20.6,       21.83,      23.12,      24.5,       25.96,      27.5,   29.14,      30.87,
            32.7,       34.65,      36.71,      38.89,      41.2,       43.65,      46.25,      49,         51.91,      55,     58.27,      61.74,
            65.41,      69.3,       73.42,      77.78,      82.41,      87.31,      92.5,       98,         103.83,     110,    116.54,     123.47,
            130.81,     138.59,     146.83,     155.56,     164.81,     174.61,     185,        196,        207.65,     220,    233.08,     246.94,
            261.63,     277.18,     293.66,     311.13,     329.63,     349.23,     369.99,     392,        415.3,      440,    466.16,     493.88,
            523.25,     554.37,     587.33,     622.25,     659.25,     698.46,     739.99,     783.99,     830.61,     880,    932.33,     987.77,
            1046.5,     1108.73,    1174.66,    1244.51,    1318.51,    1396.91,    1479.98,    1567.98,    1661.22,    1760,   1864.66,    1975.53,
            2093,       2217.46,    2349.32,    2489.02,    2637.02,    2793.83,    2959.96,    3135.96,    3322.44,    3520,   3729.31,    3951.07,
            4186.01,    4434.92,    4698.63,    4978.03,    5274.04,    5587.65,    5919.91,    6271.93,    6644.88,    7040,   7458.62,    7902.13 # technically don't need highest than 4186... I think...
        ]
