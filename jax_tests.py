import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from alif_sg.minimal_LRU_modified.lru.model import LRU


class simpleLRU(nn.Module):
    d_hidden = 1
    d_model = 1

    def __call__(self, inputs):
        print(inputs.shape)
        # outputs = jax.lax.associative_scan(jnp.add, inputs)
        outputs = jax.vmap(lambda u: jax.lax.associative_scan(jnp.add, u))(inputs)
        return outputs


batch_size = 1

time_steps = 3
features = 2
inps = jnp.arange(0, time_steps)
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)


# LRU = nn.vmap(
#     LRU,
#     in_axes=(0, 0),
#     out_axes=0,
#     variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
#     split_rngs={"params": False, "dropout": True}, axis_name='batch'
# )
#
# lru = LRU(d_model=features, d_hidden=features)
# # lru = partial(LRU, d_hidden=features, d_model=features)
# model = lru


lru_module = nn.vmap(
    LRU,
    in_axes=(None, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch'
)

lru = lru_module(d_model=features, d_hidden=features)
model = lru

jax_seed = 0
key = random.PRNGKey(jax_seed)
init_rng, train_rng = random.split(key, num=2)
dummy_input = jnp.ones((batch_size, time_steps, features))
# integration_timesteps = jnp.ones((batch_size, time_steps))
init_rng, dropout_rng = jax.random.split(init_rng, num=2)

variables = model.init(
    {"params": init_rng, "dropout": dropout_rng},
    dummy_input,
)

f = lambda x: lru(x)
# f = lambda x: simpleLRU(d_model=features, d_hidden=features)(x)

Jb = jax.vmap(lambda u: jax.jacfwd(f)(u))(inps)
Jb_one_step_back = Jb[:, :-1]
print(Jb_one_step_back.shape)
# Jb_one_step_back_no_diag = Jb_one_step_back - jnp.eye(features)[None, None, :, None, :]
# Jb_one_step_back_no_diag = Jb_one_step_back

start_time = time.perf_counter()
print(Jb.shape)
radiuses_t = []
radiuses_l = []

for i in range(time_steps - 1):
    j_t = Jb_one_step_back[:, i, :, i]
    j_l = Jb[:, i, :, i]

    # jnp maximal eigenvalue
    radius_t = jnp.linalg.norm(j_t, axis=(1, 2))
    radius_l = jnp.linalg.norm(j_l, axis=(1, 2))
    radiuses_t.append(radius_t)
    radiuses_l.append(radius_l)

radiuses_t = jnp.stack(radiuses_t, axis=0)
radiuses_l = jnp.stack(radiuses_l, axis=0)

print(f'For. Elapsed time: {time.perf_counter() - start_time:.3f} seconds.')

# Fori Loop Option
start_time = time.perf_counter()


def compute_radius(i, Jb_osb):
    j = Jb_osb[:, i, :, i]
    radius = jnp.linalg.norm(j, axis=(1, 2))
    return radius


radiuses_fori = jax.vmap(lambda i: compute_radius(i, Jb_one_step_back))(jnp.arange(time_steps - 1))

print(f'Fori Loop Elapsed time: {time.perf_counter() - start_time:.3f} seconds.')

# Compare Results
print('For Loop Results:')
print(radiuses_t.shape)
print(radiuses_t)
print(radiuses_l)

print('Fori Loop Results:')
print(radiuses_fori.shape)
