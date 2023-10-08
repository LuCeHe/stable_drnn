import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from jax.scipy.linalg import block_diag

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.S5.s5.ssm import S5SSM, init_S5SSM
from alif_sg.S5.s5.ssm_init import make_DPLR_HiPPO
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 2
time_steps = 7
features = 256
model_name = 'slru2'  # slru lru s5 slru2


class simpleLRU(nn.Module):
    d_hidden: 1
    d_model: 1

    def __call__(self, inputs):
        outputs = jax.lax.associative_scan(jnp.add, inputs)
        return outputs


def matrix_init(key, shape, dtype=jnp.float_, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class simpleLRU2(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda

    def setup(self):
        self.gamma_log = self.param("gamma_log", gamma_log_init, (self.nu_log, self.theta_log))

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
    def __call__(self, inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)

        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)

        # Use them to compute the output of the module
        outputs = jax.vmap(lambda x, u: x.real + u)(Bu_elements, inputs)

        return outputs




if model_name == 'slru':
    ssm = partial(simpleLRU, d_hidden=features, d_model=features)

if model_name == 'slru2':
    ssm = partial(simpleLRU2, d_hidden=features, d_model=features)

elif model_name == 'lru':
    ssm = partial(LRU, d_hidden=features, d_model=features)

elif model_name == 's5':
    # if s5
    blocks = 12
    ssm_size = 192
    # determine the size of initial blocks
    # ssm_size = args.ssm_size_base
    block_size = int(ssm_size / blocks)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm = init_S5SSM(H=features,
                     P=ssm_size,
                     Lambda_re_init=Lambda.real,
                     Lambda_im_init=Lambda.imag,
                     V=V,
                     Vinv=Vinv,
                     C_init='lecun_normal',
                     discretization='zoh',
                     dt_min=1.,
                     dt_max=2.,
                     conj_sym=False,
                     clip_eigs=False,
                     bidirectional=False)
else:
    raise ValueError('model_name not in [slru, lru, s5]')

BatchClassificationModel = nn.vmap(
    SequenceLayer,
    in_axes=0, out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')

layer = partial(
    BatchClassificationModel, ssm=ssm, dropout=.1, d_model=features
)(training=True)

print(layer)

inps = jnp.arange(0, time_steps)
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)

jax_seed = 0
key = random.PRNGKey(jax_seed)
init_rng, train_rng = random.split(key, num=2)
dummy_input = jnp.ones((batch_size, time_steps, features))
init_rng, dropout_rng = jax.random.split(init_rng, num=2)

variables = layer.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

params = variables["params"]
f = lambda x: layer.apply({'params': params}, x, rngs={'dropout': dropout_rng})

outs = f(inps)
print('outs.shape', outs.shape)


# print(f(inps))


# Jb = jax.vmap(lambda u: jax.jacfwd(f)(u))(inps)
# Jb =  jax.jacfwd(f)(inps)
# print('Jb.shape', Jb.shape)

# Define a function for jacfwd only with respect to the 'params' variable
def jacfwd_params_fn(x):
    print(x.shape)
    return jax.jacfwd(f)(x)


# Apply jax.jacfwd to the parameters only
Jb = jax.vmap(jacfwd_params_fn)(inps)
print('Jb.shape', Jb.shape)
