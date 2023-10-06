import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 2
time_steps = 7
features = 3


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
print(f(inps))




Jb = jax.vmap(lambda u: jax.jacfwd(f)(u))(inps)

# Jb = jax.jacfwd(f)(inps)
Jb_back = Jb[:, :-1]
print('Jb.shape', Jb.shape)

start_time = time.perf_counter()
radiuses_t = []
radiuses_l = []

for i in range(time_steps - 1):
    j_t = Jb_back[:, i, :, i]
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


def compute_radius(i, Jb_osb, Jb):
    j_t = Jb_osb[:, i, :, i]
    radius_t = jnp.linalg.norm(j_t, axis=(1, 2))

    j_l = Jb[:, i, :, i]
    radius_l = jnp.linalg.norm(j_l, axis=(1, 2))
    return radius_t, radius_l


radiuses_fori_t, radiuses_fori_l = jax.vmap(lambda i: compute_radius(i, Jb_back, Jb))(jnp.arange(time_steps - 1))

print(f'Fori Loop Elapsed time: {time.perf_counter() - start_time:.3f} seconds.')

# Compare Results
print('For Loop Results:')
print(radiuses_t.shape)
print('radiuses_t')
print(radiuses_t.shape)
print(radiuses_l.shape)

print('Fori Loop Results:')
print(radiuses_fori_t.shape)
print(radiuses_fori_l.shape)


print('Check if the results are the same:')
print(jnp.allclose(radiuses_t, radiuses_fori_t))
print(jnp.allclose(radiuses_l, radiuses_fori_l))

print(radiuses_t)
print(radiuses_l)