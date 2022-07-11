import multiprocessing

import tensorflow as tf
import numpy as np
import tensorflow_io as tfio

from deepafx.tf2lib.specs import spectral_ops


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


# def numpy_load_abs(x):
#     x = np.abs(np.load(x))
#     x = x.astype('float32')
#     return x

# @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
# def tf_np_load_abs(input):
#     y = tf.numpy_function(numpy_load_abs, [input], tf.float32)
#     return y

@tf.function
def get_spectrogram(waveform, audio_length):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros(audio_length - waveform.shape[0], dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length,
                                 frame_length=512,
                                 frame_step=256,
                                 pad_end=True)
    spectrogram = tf.abs(spectrogram)
    return spectrogram


@tf.function
def _safe_log(x):
  return tf.math.log(x + 1.0e-6)


@tf.function
def _linear_to_mel_matrix(sr=16000,nfft=512, mel_downscale=2):
    """Get the mel transformation matrix."""
    num_freq_bins = (nfft // 2)+1
    lower_edge_hertz = 0.0
    upper_edge_hertz = sr / 2.0
    num_mel_bins = num_freq_bins // mel_downscale
    return spectral_ops.linear_to_mel_weight_matrix(num_mel_bins, 
                                                    num_freq_bins, 
                                                    sr, 
                                                    lower_edge_hertz,
                                                    upper_edge_hertz)

@tf.function
def get_log_mel_spec(stfts):
    logmag = _safe_log(tf.abs(stfts))
    mag2 = tf.exp(2.0 * logmag)
    
    l2mel = tf.cast(_linear_to_mel_matrix(),dtype='float32')
    logmelmag2 = _safe_log(tf.tensordot(mag2, l2mel, 1))
    return logmelmag2





# @tf.function
# def resample_audio(audio):
#     return tfio.audio.resample(audio, 44100, 16000)

def disk_image_batch_dataset(img_paths,
                             batch_size,
                             length,
                             labels=None,
                             representation='logmel',
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             filter_fn=None,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """Batch dataset of disk image for PNG and JPEG.

    Parameters
    ----------
    img_paths : 1d-tensor/ndarray/list of str
    labels : nested structure of tensors/ndarrays/lists

    to do: change the fixed downsamping method

    """
    if labels is None:
        memory_data = img_paths
    else:
        memory_data = (img_paths, labels)
    

    def parse_fn(path, *label):

        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, desired_channels=1,
                                  desired_samples=length, name=None)



        audio = tf.squeeze(audio, axis=-1)
        # audio = tf.expand_dims(audio,0)


        if representation == 'magspec':
            spectrogram = get_spectrogram(audio, length)
            spectrogram = tf.expand_dims(spectrogram, -1)
        
        
        if representation == 'logmel':
            spectrogram = get_spectrogram(audio, length)
            spectrogram = get_log_mel_spec(spectrogram)
            spectrogram = tf.expand_dims(spectrogram, -1)
        
        # if representation == 'hcnn':
        #     spectrogram = get_filtered_specs(spectrogram)

        
            

        # spectrogram = spectrogram[:257,:517,:]   #make variable spec size

        return (spectrogram,) + label

        # return (spectrogram,) + label
        # else:
        #     print(spectrogram.shape)






    if map_fn:  # fuse `map_fn` and `parse_fn`
        def map_fn_(*args):
            return map_fn(*parse_fn(*args))
    else:
        map_fn_ = parse_fn

    dataset = memory_data_batch_dataset(memory_data,
                                        batch_size,
                                        drop_remainder=drop_remainder,
                                        n_prefetch_batch=n_prefetch_batch,
                                        filter_fn=filter_fn,
                                        map_fn=map_fn_,
                                        n_map_threads=n_map_threads,
                                        filter_after_map=filter_after_map,
                                        shuffle=shuffle,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat)

    return dataset



