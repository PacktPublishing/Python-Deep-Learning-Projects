import json
import os
import math
import time
import argparse
from datetime import datetime
import deepSpeech
import numpy as np
import tensorflow as tf
from Levenshtein import distance

# Note this definition must match the ALPHABET chosen in
# preprocess_Librispeech.py
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
IX_TO_CHAR = {i: ch for (i, ch) in enumerate(ALPHABET)}


def parse_args():
    """ Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', type=str,
                        default='../models/librispeech/eval',
                        help='Directory to write event logs')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='../models/librispeech/train',
                        help='Directory where to read model checkpoints.')
    parser.add_argument('--eval_data', type=str, default='val',
                        help="Either 'test' or 'val' or 'train' ")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of feats to process in a batch')
    parser.add_argument('--eval_interval_secs', type=int, default=60 * 5,
                        help='How often to run the eval')
    parser.add_argument('--data_dir', type=str,
                        default='../data/librispeech/processed/',
                        help='Path to the deepSpeech data directory')
    parser.add_argument('--run_once', type=bool, default=False,
                        help='Whether to run eval only once')
    args = parser.parse_args()

    # Read saved parameters from file
    param_file = os.path.join(args.checkpoint_dir,
                              'deepSpeech_parameters.json')
    with open(param_file, 'r') as file:
        params = json.load(file)
        # Read network architecture parameters from
        # previously saved parameter file.
        args.num_hidden = params['num_hidden']
        args.num_rnn_layers = params['num_rnn_layers']
        args.rnn_type = params['rnn_type']
        args.num_filters = params['num_filters']
        args.use_fp16 = params['use_fp16']
        args.temporal_stride = params['temporal_stride']
        args.moving_avg_decay = params['moving_avg_decay']
    return args


def sparse_to_labels(sparse_matrix):
    """ Convert index based transcripts to strings"""
    results = ['']*sparse_matrix.dense_shape[0]
    for i, val in enumerate(sparse_matrix.values.tolist()):
        results[sparse_matrix.indices[i, 0]] += IX_TO_CHAR[val]
    return results


def initialize_from_checkpoint(sess, saver):
    """ Initialize variables on the graph"""

    # Initialise variables from a checkpoint file, if provided.
    ckpt = tf.train.get_checkpoint_state(ARGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/train/model.ckpt-0,
        # extract global_step from it.
        checkpoint_path = ckpt.model_checkpoint_path
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        return global_step
    else:
        print('No checkpoint file found')
        return


def inference(predictions_op, true_labels_op, display, sess):
    """ Perform inference per batch on pre-trained model.
    This function performs inference and computes the CER per utterance.
    Args:
        predictions_op: Prediction op
        true_labels_op: True Labels op
        display: print sample predictions if True
        sess: default session to evaluate the ops.
    Returns:
        char_err_rate: list of CER per utterance.
    """
    char_err_rate = []
    # Perform inference of batch worth of data at a time.
    [predictions, true_labels] = sess.run([predictions_op,
                                           true_labels_op])
    pred_label = sparse_to_labels(predictions[0][0])
    actual_label = sparse_to_labels(true_labels)
    for (label, pred) in zip(actual_label, pred_label):
        char_err_rate.append(distance(label, pred)/len(label))

    if display:
        # Print sample responses
        for i in range(ARGS.batch_size):
            print(actual_label[i] + ' vs ' + pred_label[i])
    return char_err_rate


def eval_once(saver, summary_writer, predictions_op, summary_op,
              true_labels_op):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      predictions_ops: Op to compute predictions.
      summary_op: Summary op.
    """
    with tf.Session() as sess:

        # Initialize weights from checkpoint file.
        global_step = initialize_from_checkpoint(sess, saver)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for queue_runners in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue_runners.create_threads(sess, coord=coord,
                                                            daemon=True,
                                                            start=True))
            # Only using a subset of the training data
            if ARGS.eval_data == 'train':
                num_examples = 2048

            elif ARGS.eval_data == 'val':
                num_examples = 2703

            elif ARGS.eval_data == 'test':
                num_examples = 2620
            num_iter = int(math.ceil(num_examples / ARGS.batch_size))
            step = 0
            char_err_rate = []
            while step < num_iter and not coord.should_stop():
                char_err_rate.append(inference(predictions_op, true_labels_op,
                                               step == 0, sess))
                step += 1

            # Compute and print mean CER
            avg_cer = np.mean(char_err_rate)*100
            print('%s: char_err_rate = %.3f %%' % (datetime.now(), avg_cer))

            # Add summary ops
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='char_err_rate', simple_value=avg_cer)
            summary_writer.add_summary(summary, global_step)
        except Exception as exc:  # pylint: disable=broad-except
            coord.request_stop(exc)

        # Close threads
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """ Evaluate deepSpeech modelfor a number of steps."""

    with tf.Graph().as_default() as graph:

        # Get feats and labels for deepSpeech.
        feats, labels, seq_lens = deepSpeech.inputs(ARGS.eval_data,
                                                    data_dir=ARGS.data_dir,
                                                    batch_size=ARGS.batch_size,
                                                    use_fp16=ARGS.use_fp16,
                                                    shuffle=True)

        # Build ops that computes the logits predictions from the
        # inference model.
        ARGS.keep_prob = 1.0  # Disable dropout during testing.
        logits = deepSpeech.inference(feats, seq_lens, ARGS)

        # Calculate predictions.
        output_log_prob = tf.nn.log_softmax(logits)
        decoder = tf.nn.ctc_greedy_decoder
        strided_seq_lens = tf.div(seq_lens, ARGS.temporal_stride)
        predictions = decoder(output_log_prob, strided_seq_lens)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            ARGS.moving_avg_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(ARGS.eval_dir, graph)

        while True:
            eval_once(saver, summary_writer, predictions, summary_op, labels)

            if ARGS.run_once:
                break
            time.sleep(ARGS.eval_interval_secs)


def main():
    """
    Create eval directory and perform inference on checkpointed model.
    """
    if tf.gfile.Exists(ARGS.eval_dir):
        tf.gfile.DeleteRecursively(ARGS.eval_dir)
    tf.gfile.MakeDirs(ARGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    ARGS = parse_args()
    main()
