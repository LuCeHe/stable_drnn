from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def normal_init(key, shape, dtype=jnp.float_, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def uniform_init(key, shape, dtype=jnp.float_, normalization=1):
    return jax.random.uniform(key=key, shape=shape, dtype=dtype, minval=-1, maxval=1) / normalization


class LRU_real(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions

    def setup(self):
        self.diag_lambda = self.param(
            "nu_log", partial(uniform_init, normalization=jnp.sqrt(self.d_model)),
            (self.d_hidden,)
        )

        # Glorot initialized Input/Output projection matrices
        self.B = self.param(
            "B", partial(normal_init, normalization=jnp.sqrt(self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C = self.param(
            "C", partial(normal_init, normalization=jnp.sqrt(self.d_hidden)),
            (self.d_model, self.d_hidden),
        )

        self.D = self.param("D", partial(normal_init, normalization=1), (self.d_model,))

    def __call__(self, inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""

        Lambda_elements = jnp.repeat(self.diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: self.B @ u)(inputs)

        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))

        # Use them to compute the output of the module
        outputs = jax.vmap(lambda x, u: self.C @ x + self.D * u)(hidden_states, inputs)

        return outputs


class twolru(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    # lru1: LRU_real  # lru module
    # lru2: LRU_real  # lru module
    d_model: int  # model size
    d_hidden: int  # hidden state size

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq1 = LRU_real(d_hidden=self.d_hidden, d_model=self.d_model)
        self.seq2 = LRU_real(d_hidden=self.d_hidden, d_model=self.d_model)

    def __call__(self, inputs):
        z = self.seq1(inputs) + inputs  # call LRU
        y = self.seq2(inputs)
        return z * y


if __name__ == '__main__':
    pass
