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
    comments: str = ''  # comments for the module

    def setup(self):
        if '2m1emeldiag' in self.comments:
            from alif_sg.S5.s5.lru_model import nu_init
            self.diag_lambda = self.param(
                "diag_lambda", partial(nu_init, r_min=.01, r_max=.99), (self.d_hidden,)
            )
        elif 'emeldiag' in self.comments:
            from alif_sg.S5.s5.lru_model import nu_init
            self.diag_lambda = self.param(
                "diag_lambda", partial(nu_init, r_min=.4, r_max=.99), (self.d_hidden,)
            )
        else:
            self.diag_lambda = self.param(
                "diag_lambda", partial(uniform_init, normalization=jnp.sqrt(self.d_model)),
                (self.d_hidden,)
            )

        self.rho=1
        if 'balancep5' in self.comments:
            self.rho = self.param(
                "rho", partial(nu_init, r_min=.01, r_max=.99), (self.d_hidden,)
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
        if '2m1emeldiag' in self.comments:
            diag_lambda = 2 * jnp.exp(-jnp.exp(self.diag_lambda)) - 1
        elif 'emeldiag' in self.comments:
            diag_lambda = jnp.exp(-jnp.exp(self.diag_lambda))
        else:
            diag_lambda = self.diag_lambda

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)

        B = self.B
        if 'emeb' in self.comments:
            B = (2 * jnp.exp(-jnp.exp(B)) - 1) * jnp.sqrt(3 / self.d_model)

        Bu_elements = jax.vmap(lambda u: B @ u)(inputs)

        rho, mrho = 1, 1
        if 'balancep5' in self.comments:
            rho = jnp.exp(-jnp.exp(self.rho))
            mrho = 1 - rho

        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (rho*Lambda_elements, mrho*Bu_elements))

        # Use them to compute the output of the module
        outputs = jax.vmap(lambda x, u: self.C @ x + self.D * u)(hidden_states, inputs)

        return outputs


class twolru(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    # lru1: LRU_real  # lru module
    # lru2: LRU_real  # lru module
    d_model: int  # model size
    d_hidden: int  # hidden state size
    comments: str = ''  # comments for the module

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq1 = LRU_real(d_hidden=self.d_hidden, d_model=self.d_model, comments=self.comments)
        self.seq2 = LRU_real(d_hidden=self.d_hidden, d_model=self.d_model, comments=self.comments)

    def __call__(self, inputs):
        z = self.seq1(inputs) + inputs  # call LRU
        y = self.seq2(inputs)
        return z * y


if __name__ == '__main__':
    pass
