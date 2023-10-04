import jax
import jax.ops
import jax.numpy as jnp
# import numpy as onp  # convention: original numpy

import flax
from flax import linen as nn


class LRU(nn.Module):

    def __call__(self, inputs):
        # outputs = jax.lax.associative_scan(jnp.add, inputs)
        outputs = jax.vmap(lambda u: jax.lax.associative_scan(jnp.add, u))(inputs)
        return outputs


inps = jnp.arange(0, 4)

batch_size = 1
features = 1
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)

outs = LRU()(inps)

print('inps')
print(inps)
print('outs')
print(outs)
d2fdx = jax.grad(LRU())
print(d2fdx(inps))
