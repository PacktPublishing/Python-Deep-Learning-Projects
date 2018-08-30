import tensorflow as tf
import DS_input as deepSpeech_input
import rnn_cell
from helper_routines import _variable_on_cpu
from helper_routines import _variable_with_weight_decay
from helper_routines import _activation_summary

# Global constants describing the speech data set.
NUM_CLASSES = deepSpeech_input.NUM_CLASSES
NUM_PER_EPOCH_FOR_TRAIN = deepSpeech_input.NUM_PER_EPOCH_FOR_TRAIN
NUM_PER_EPOCH_FOR_EVAL = deepSpeech_input.NUM_PER_EPOCH_FOR_EVAL
NUM_PER_EPOCH_FOR_TEST = deepSpeech_input.NUM_PER_EPOCH_FOR_TEST


def inputs(eval_data, data_dir, batch_size, use_fp16, shuffle):
    """Construct input for LibriSpeech model evaluation using the Reader ops.

    Args:
      eval_data: 'train', 'test' or 'eval'
      data_dir: folder containing the pre-processed data
      batch_size: int,size of mini-batch
      use_fp16: bool, if True use fp16 else fp32
      shuffle: bool, to shuffle the tfrecords or not.

    Returns:
      feats: MFCC. 4D tensor of [batch_size, T, F, 1] size.
      labels: Labels. 1D tensor of [batch_size] size.
      seq_lens: SeqLens. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    feats, labels, seq_lens = deepSpeech_input.inputs(eval_data=eval_data,
                                                      data_dir=data_dir,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)
    if use_fp16:
        feats = tf.cast(feats, tf.float16)
    return feats, labels, seq_lens


def inference(feats, seq_lens, params):
    """Build the deepSpeech model.

    Args:
      feats: MFCC features returned from distorted_inputs() or inputs().
      seq_lens: Input sequence length per utterance.
      params: parameters of the model.

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU
    # training runs. If we only ran this model on a single GPU,
    # we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().

    if params.use_fp16:
        dtype = tf.float16
    else:
        dtype = tf.float32

    feat_len = feats.get_shape().as_list()[-1]

    # convolutional layers
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[11, feat_len, 1, params.num_filters],
            wd_value=None, use_fp16=params.use_fp16)

        feats = tf.expand_dims(feats, dim=-1)
        conv = tf.nn.conv2d(feats, kernel,
                            [1, params.temporal_stride, 1, 1],
                            padding='SAME')
        # conv = tf.nn.atrous_conv2d(feats, kernel, rate=2, padding='SAME')
        biases = _variable_on_cpu('biases', [params.num_filters],
                                  tf.constant_initializer(-0.05),
                                  params.use_fp16)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

        # dropout
        conv1_drop = tf.nn.dropout(conv1, params.keep_prob)

    # recurrent layers
    with tf.variable_scope('rnn') as scope:

        # Reshape conv output to fit rnn input
        rnn_input = tf.reshape(conv1_drop, [params.batch_size, -1,
                                            feat_len*params.num_filters])
        # Permute into time major order for rnn
        rnn_input = tf.transpose(rnn_input, perm=[1, 0, 2])
        # Make one instance of cell on a fixed device,
        # and use copies of the weights on other devices.
        cell = rnn_cell.CustomRNNCell(
            params.num_hidden, activation=tf.nn.relu6,
            use_fp16=params.use_fp16)
        drop_cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=params.keep_prob)
        multi_cell = tf.contrib.rnn.MultiRNNCell(
            [drop_cell] * params.num_rnn_layers)

        seq_lens = tf.div(seq_lens, params.temporal_stride)
        if params.rnn_type == 'uni-dir':
            rnn_outputs, _ = tf.nn.dynamic_rnn(multi_cell, rnn_input,
                                               sequence_length=seq_lens,
                                               dtype=dtype, time_major=True,
                                               scope='rnn',
                                               swap_memory=True)
        else:
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                multi_cell, multi_cell, rnn_input,
                sequence_length=seq_lens, dtype=dtype,
                time_major=True, scope='rnn',
                swap_memory=True)
            outputs_fw, outputs_bw = outputs
            rnn_outputs = outputs_fw + outputs_bw
        _activation_summary(rnn_outputs)

    # Linear layer(WX + b) - softmax is applied by CTC cost function.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay(
            'weights', [params.num_hidden, NUM_CLASSES],
            wd_value=None,
            use_fp16=params.use_fp16)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0),
                                  params.use_fp16)
        logit_inputs = tf.reshape(rnn_outputs, [-1, cell.output_size])
        logits = tf.add(tf.matmul(logit_inputs, weights),
                        biases, name=scope.name)
        logits = tf.reshape(logits, [-1, params.batch_size, NUM_CLASSES])
        _activation_summary(logits)

    return logits


def loss(logits, labels, seq_lens):
    """Compute mean CTC Loss.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
      seq_lens: Length of each utterance for ctc cost computation.

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average ctc loss across the batch.
    ctc_loss = tf.nn.ctc_loss(inputs=tf.cast(logits, tf.float32),
                              labels=labels, sequence_length=seq_lens)
    ctc_loss_mean = tf.reduce_mean(ctc_loss, name='ctc_loss')
    tf.add_to_collection('losses', ctc_loss_mean)

    # The total loss is defined as the cross entropy loss plus all
    # of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in deepSpeech model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for each_loss in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average
        # version of the loss as the original loss name.
        tf.scalar_summary(each_loss.op.name + ' (raw)', each_loss)
        tf.scalar_summary(each_loss.op.name, loss_averages.average(each_loss))

    return loss_averages_op
