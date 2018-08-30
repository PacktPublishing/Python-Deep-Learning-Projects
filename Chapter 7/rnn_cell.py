"""
Custom RNN Cell definition.
Default RNNCell in TensorFlow throws errors when
variables are re-used between devices.
"""

from tensorflow.contrib import rnn as rnn_cell
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import tensorflow as tf


def _variable_on_cpu(name, shape, initializer=None, use_fp16=False):
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
        var = tf.get_variable(name, shape=shape, initializer=initializer,
                              dtype=dtype)
    return var


class CustomRNNCell(rnn_cell.BasicRNNCell):
    """ This is a custoRNNCell that allows the weights
    to be re-used on multiple devices. In particular, the Matrix of weights is
    set using _variable_on_cpu.
    The default version of the BasicRNNCell, did not support the ability to
    pin weights on one device (say cpu).
    """

    def __init__(self, num_units, activation=tanh, use_fp16=False):
        self._num_units = num_units
        self._activation = activation
        self.use_fp16 = use_fp16

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN:
        output = new_state = activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            output = self._activation(_linear([inputs, state], self._num_units,
                                              True, use_fp16=self.use_fp16))
        return output, output


def _linear(args, output_size, bias, scope=None, use_fp16=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = _variable_on_cpu('Matrix', [total_arg_size, output_size],
                                  use_fp16=use_fp16)
        if use_fp16:
            dtype = tf.float16
        else:
            dtype = tf.float32
        args = [tf.cast(x, dtype) for x in args]
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = _variable_on_cpu('Bias', [output_size],
                                     tf.constant_initializer(0),
                                     use_fp16=use_fp16)
    return res + bias_term
