import os.path
import glob
import tensorflow as tf

# Global constants describing the dataset
# Note this definition must match the ALPHABET chosen in
# preprocess_Librispeech.py
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "  # for LibriSpeech
NUM_CLASSES = len(ALPHABET) + 1  # Additional class for blank character
NUM_PER_EPOCH_FOR_TRAIN = 28535
NUM_PER_EPOCH_FOR_EVAL = 2703
NUM_PER_EPOCH_FOR_TEST = 2620


def _generate_feats_and_label_batch(filename_queue, batch_size):
    """Construct a queued batch of spectral features and transcriptions.

    Args:
      filename_queue: queue of filenames to read data from.
      batch_size: Number of utterances per batch.

    Returns:
      feats: mfccs. 4D tensor of [batch_size, height, width, 3] size.
      labels: transcripts. List of length batch_size.
      seq_lens: Sequence Lengths. List of length batch_size.
    """

    # Define how to parse the example
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
        "labels": tf.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        # mfcc features are 13 dimensional
        "feats": tf.FixedLenSequenceFeature([13, ], dtype=tf.float32)
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # Generate a batch worth of examples after bucketing
    seq_len, (feats, labels) = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.cast(context_parsed['seq_len'], tf.int32),
        tensors=[sequence_parsed['feats'], context_parsed['labels']],
        batch_size=batch_size,
        bucket_boundaries=list(range(100, 1900, 100)),
        allow_smaller_final_batch=True,
        num_threads=16,
        dynamic_pad=True)

    return feats, tf.cast(labels, tf.int32), seq_len


def inputs(eval_data, data_dir, batch_size, shuffle=False):
    """Construct input for fordspeech evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the fordspeech data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of
              [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if eval_data == 'train':
        num_files = len(glob.glob(os.path.join(data_dir,
                                               'train*/*.tfrecords')))
        filenames = [os.path.join(data_dir, 'train-clean-100/train_' +
                                  str(i) + '.tfrecords')
                     for i in range(1, num_files+1)]
    elif eval_data == 'val':
        filenames = glob.glob(os.path.join(data_dir, 'dev*/*.tfrecords'))

    elif eval_data == 'test':
        filenames = glob.glob(os.path.join(data_dir, 'test*/*.tfrecords'))

    for file in filenames:
        if not tf.gfile.Exists(file):
            raise ValueError('Failed to find file: ' + file)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_feats_and_label_batch(filename_queue, batch_size)
