import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from jax.scipy.linalg import block_diag

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.S5.s5.ssm import S5SSM
from alif_sg.S5.s5.ssm_init import make_DPLR_HiPPO
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 2
time_steps = 7
features = 3
model_name = 's5' # slru lru s5

class simpleLRU(nn.Module):
    d_hidden: 1
    d_model: 1

    def __call__(self, inputs):
        outputs = jax.lax.associative_scan(jnp.add, inputs)
        return outputs


if model_name == 'slru':
    ssm = partial(simpleLRU, d_hidden=features, d_model=features)

elif model_name == 'lru':
    ssm = partial(LRU, d_hidden=features, d_model=features)

elif model_name == 's5':
    # if s5
    blocks = 2
    block_size = int(features / blocks)

    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))
    ssm = partial(
        S5SSM,
        H=features, P=features,
        Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag, V=V, Vinv=Vinv,
        dt_min=1, dt_max=2, C_init='lecun_normal', discretization='zoh'
    )

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
