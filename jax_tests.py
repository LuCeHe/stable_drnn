import jax
import jax.ops
import jax.numpy as jnp
# import numpy as onp  # convention: original numpy

import flax
from flax import linen as nn


class LRU(nn.Module):

    def __call__(self, inputs):
        outputs = jax.lax.associative_scan(jnp.add, inputs)
        return outputs


inps = jnp.arange(0, 4)
inps = jnp.float32(inps)

outs = LRU()(inps)

print(outs)
d2fdx = jax.grad(LRU())
print(d2fdx(inps))
