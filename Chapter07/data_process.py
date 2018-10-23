"""Creates SequenceExamples and stores them in TFRecords format.
Computes spectral features from raw audio waveforms and groups the audio into
multiple TFRecords files based on their length. The utterances are stored in
sorted order based on length to allow for sorta-grad implementation.
Note:
This script can take a few hours to run to compute and store the mfcc
features on the 100 hour Librispeech dataset.
"""
import os
import glob
import soundfile as sf
from python_speech_features import mfcc
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_mfcc(audio_data, sample_rate):
    ''' Computes the mel-frequency cepstral coefficients.
    The audio time series is normalised and its mfcc features are computed.
    Args:
        audio_data: time series of the speech utterance.
        sample_rate: sampling rate.
    Returns:
        mfcc_feat:[num_frames x F] matrix representing the mfcc.
    '''

    audio_data = audio_data - np.mean(audio_data)
    audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.025, winstep=0.01,
                     numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat


def make_example(seq_len, spec_feat, labels):
    ''' Creates a SequenceExample for a single utterance.
    This function makes a SequenceExample given the sequence length,
    mfcc features and corresponding transcript.
    These sequence examples are read using tf.parse_single_sequence_example
    during training.
    Note: Some of the tf modules used in this function(such as
    tf.train.Feature) do not have comprehensive documentation in v0.12.
    This function was put together using the test routines in the
    tensorflow repo.
    See: https://github.com/tensorflow/tensorflow/
    blob/246a3724f5406b357aefcad561407720f5ccb5dc/
    tensorflow/python/kernel_tests/parsing_ops_test.py
    Args:
        seq_len: integer represents the sequence length in time frames.
        spec_feat: [TxF] matrix of mfcc features.
        labels: list of ints representing the encoded transcript.
    Returns:
        Serialized sequence example.
    '''
    # Feature lists for the sequential features of the example
    feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                  for frame in spec_feat]
    feat_dict = {"feats": tf.train.FeatureList(feature=feats_list)}
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    # Context features for the entire sequence
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))

    context_feats = tf.train.Features(feature={"seq_len": len_feat,
                                               "labels": label_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()


def process_data(partition):
    """ Reads audio waveform and transcripts from a dataset partition
    and generates mfcc featues.
    Args:
        parition - represents the dataset partition name.
    Returns:
        feats: dict containing mfcc feature per utterance
        transcripts: dict of lists representing transcript.
        utt_len: dict of ints holding sequence length of each
                 utterance in time frames.
    """

    feats = {}
    transcripts = {}
    utt_len = {}  # Required for sorting the utterances based on length

    for filename in glob.iglob(partition+'/**/*.txt', recursive=True):
        with open(filename, 'r') as file:
            for line in file:
                parts = line.split()
                audio_file = parts[0]
                file_path = os.path.join(os.path.dirname(filename),
                                         audio_file+'.flac')
                audio, sample_rate = sf.read(file_path)                
                feats[audio_file] = compute_mfcc(audio, sample_rate)
                plot_audio(audio,feats[audio_file])
                utt_len[audio_file] = feats[audio_file].shape[0]
                target = ' '.join(parts[1:])
                transcripts[audio_file] = [CHAR_TO_IX[i] for i in target]
    return feats, transcripts, utt_len

def plot_audio(audio , mfcc):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs[0].plot(audio);
    axs[0].set_title('Raw Audio Signal')
    axs[1].plot(mfcc);
    axs[1].set_title('mel-frequency cepstral coefficients')

def create_records():
    """ Pre-processes the raw audio and generates TFRecords.
    This function computes the mfcc features, encodes string transcripts
    into integers, and generates sequence examples for each utterance.
    Multiple sequence records are then written into TFRecord files.
    """
    for partition in sorted(glob.glob(AUDIO_PATH+'/*')):
        print('Processing' + partition)
        feats, transcripts, utt_len = process_data(partition)
        sorted_utts = sorted(utt_len, key=utt_len.get)
        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]]/100)
        min_t = int(utt_len[sorted_utts[0]]/100)

        # Create destination directory
        write_dir = '../data/librispeech/processed/' + partition.split('/')[-1]
        if tf.gfile.Exists(write_dir):
            tf.gfile.DeleteRecursively(write_dir)
        tf.gfile.MakeDirs(write_dir)

        if os.path.basename(partition) == 'train-clean-100':
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t+1):
                filename = os.path.join(write_dir, 'train' + '_' + str(i) +
                                        '.tfrecords')
                writer[i] = tf.python_io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts):
                example = make_example(utt_len[utt], feats[utt].tolist(),
                                       transcripts[utt])
                index = int(utt_len[utt]/100)
                writer[index].write(example)
                count[index] += 1

            for i in range(min_t, max_t+1):
                writer[i].close()
            print(count)

            # Remove bins which have fewer than 20 utterances
            for i in range(min_t, max_t+1):
                if count[i] < 20:
                    os.remove(os.path.join(write_dir, 'train' +
                                           '_' + str(i) + '.tfrecords'))
        else:
            # Create single TFRecord for dev and test partition
            filename = os.path.join(write_dir, os.path.basename(write_dir) +
                                    '.tfrecords')
            print('Creating', filename)
            record_writer = tf.python_io.TFRecordWriter(filename)
            for utt in sorted_utts:
                example = make_example(utt_len[utt], feats[utt].tolist(),
                                       transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed '+str(len(sorted_utts))+' audio files')

# Audio path is the location of the directory that contains the librispeech
# data partitioned into three folders: dev-clean, train-clean-100, test-clean
AUDIO_PATH = '../data/librispeech/audio/'
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
CHAR_TO_IX = {ch: i for (i, ch) in enumerate(ALPHABET)}

if __name__ == '__main__':
    create_records()
