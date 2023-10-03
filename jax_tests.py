
import jax
import jax.ops
import jax.numpy as np
import numpy as onp  # convention: original numpy

import flax
from flax import linen as nn
from flax import optim

class RNNCell(nn.Module):
  @nn.compact
  def __call__(self, state, x):
    # Wh @ h + Wx @ x + b can be efficiently computed
    # by concatenating the vectors and then having a single dense layer
    x = np.concatenate([state, x])
    new_state = np.tanh(nn.Dense(state.shape[0])(x))
    return new_state

def one_hot(i, n):
  """
  create vector of size n with 1 at index i
  """
  x = np.zeros(n)
  return jax.ops.index_update(x, i, 1)

def encode(char):
  return one_hot(char_to_id[char], len(char_to_id))

def decode(predictions, id_to_char):
  # for simplicity, pick the most likely character
  # this can be replaced by sampling weighted
  # by the probability of each character
  return id_to_char[int(np.argmax(predictions))]

class ChaRNN(nn.Module):
  state_size: int
  vocab_size: int

  @nn.compact
  def __call__(self, state, i):
    x = one_hot(i, self.vocab_size)
    new_state = []

    # a rather naive way of stacking multiple RNN cells
    new_state_1 = RNNCell()(state[0], x)
    new_state_2 = RNNCell()(state[1], new_state_1)
    new_state_3 = RNNCell()(state[2], new_state_2)
    predictions = nn.softmax(nn.Dense(self.vocab_size)(new_state_3))
    return [new_state_1, new_state_2, new_state_3], predictions

  def init_state(self):
    # a convenient way to initialize the state
    return [
      np.zeros(self.state_size),
      np.zeros(self.state_size),
      np.zeros(self.state_size)
    ]