import tensorflow as tf


@tf.function
def hz_to_midi(hz):
    return 12 * (log2(hz) - log2(440.0)) + 69


@tf.function
def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0)/12.0))


@tf.function
def log2(x):
    numerator = tf.math.log(tf.cast(x,tf.float32))
    denominator = tf.math.log(2.0)
    return numerator / denominator


@tf.function
def initialize_filterbank(sample_rate, n_harmonic,semitone_scale):
    # MIDI
    # lowest note (C1)
    low_midi = 24.0

    # highest note
    high_midi = tf.math.round(hz_to_midi(sample_rate / (2 * n_harmonic))) #check rounding

    # number of scales
    level = tf.cast((high_midi - low_midi) * semitone_scale,tf.int32)
    midi = tf.linspace(low_midi, high_midi, level + 1)
    hz = midi_to_hz(midi[:-1])


    # stack harmonics
    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = tf.concat((harmonic_hz, hz * (i+1)),0)




    return harmonic_hz, level


# def amplitude_to_db(amplitude, use_tf=True):
#       """Converts amplitude to decibels."""
#       lib = tf if use_tf else np
#       log10 = (lambda x: tf.math.log(x) / tf.math.log(10.0)) if use_tf else np.log10
#       amin = 1e-20  # Avoid log(0) instabilities.
#       db = log10(lib.maximum(amin, amplitude))
#       db *= 20.0
#       return db



# @tf.function <<<<< why not? is this slowing it down

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(tf.cast(x,tf.float32)) 
        # numerator = tf.math.log((x))
        denominator = tf.math.log(tf.constant(10.0, dtype=numerator.dtype))
        return numerator / denominator
    

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

