import re
import tensorflow as tf


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(act):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      act: Tensor
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', act.op.name)
    tf.summary.histogram(tensor_name + '/activations', act)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(act))


def _variable_on_cpu(name, shape, initializer, use_fp16):
    """Helper to create a Variable stored on cpu memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape,
                              initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd_value, use_fp16):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                       mode='FAN_IN',
                                                       uniform=False,
                                                       seed=None,
                                                       dtype=dtype), use_fp16)
    if wd_value is not None:
        weight_decay = tf.cast(tf.mul(tf.nn.l2_loss(var),
                                      wd_value, name='weight_loss'),
                               tf.float32)  # CTC loss is in float32
        tf.add_to_collection('losses', weight_decay)
    return var
