import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 1
time_steps = 3
features = 2


class simpleLRU(nn.Module):
    d_hidden: 1
    d_model: 1

    def __call__(self, inputs):
        print(inputs.shape)
        outputs = jax.lax.associative_scan(jnp.add, inputs)
        # outputs = jax.vmap(lambda u: jax.lax.associative_scan(jnp.add, u))(inputs)
        return outputs


ssm = partial(simpleLRU, d_hidden=features, d_model=features)
layer = SequenceLayer(
    ssm=ssm,
    dropout=.1,
    d_model=features,
)

inps = jnp.arange(0, time_steps)
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)



# lru_module = nn.vmap(
#     simpleLRU,
#     in_axes=(0,),
#     out_axes=0,
#     variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
#     split_rngs={"params": False, "dropout": True}, axis_name='batch')
#
# model = lru_module(d_hidden=features, d_model=features)

jax_seed = 0
key = random.PRNGKey(jax_seed)
init_rng, train_rng = random.split(key, num=2)
dummy_input = jnp.ones((batch_size, time_steps, features))
init_rng, dropout_rng = jax.random.split(init_rng, num=2)

variables = layer.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

f = lambda x: layer.apply(x)
print(f(inps))
