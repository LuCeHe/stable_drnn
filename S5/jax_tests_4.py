import time
from functools import partial

import jax
import jax.ops
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from flax import linen as nn
from jax.scipy.linalg import block_diag

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.S5.s5.ssm import S5SSM, init_S5SSM
from alif_sg.S5.s5.ssm_init import make_DPLR_HiPPO
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 2
time_steps = 7
features = 3

ssm = partial(LRU, d_hidden=features, d_model=features)

BatchClassificationModel = nn.vmap(
    SequenceLayer,
    in_axes=0, out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')

model = partial(
    BatchClassificationModel, ssm=ssm, dropout=.1, d_model=features
)(training=True)

print(model)

inps = jnp.arange(0, time_steps)
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)

jax_seed = 0
key = random.PRNGKey(jax_seed)
init_rng, train_rng = random.split(key, num=2)
dummy_input = jnp.ones((batch_size, time_steps, features))
init_rng, dropout_rng = jax.random.split(init_rng, num=2)

variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

params = variables["params"]


def get_radiuses(model, params, dropout_rng):
    f = lambda x: model.apply({'params': params}, x, rngs={'dropout': dropout_rng})

    # calculate the jacobian
    Jb = jax.jacfwd(f)(inps)

    # remove cross batch elements
    Jb = jnp.diagonal(Jb, axis1=0, axis2=3)
    Jb = jnp.moveaxis(Jb, [-1, ], [0, ])

    # remove the first time step
    Jb_back = Jb[:, 1:]

    def compute_radius(i, Jb_osb, Jb):
        j_t = Jb_osb[:, i, :, i]
        radius_t = jnp.linalg.norm(j_t, axis=(1, 2))

        j_l = Jb[:, i, :, i]
        radius_l = jnp.linalg.norm(j_l, axis=(1, 2))
        return radius_t, radius_l

    radiuses_t, radiuses_l = jax.vmap(lambda i: compute_radius(i, Jb_back, Jb))(jnp.arange(time_steps - 1))
    return radiuses_t, radiuses_l


target_norm = 1

# TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, inputs, labels):
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, inputs)
        loss = jnp.mean((outputs - target_norm) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.update(grads=grads)

    return new_state, loss


state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx,
)
for batch in ds.as_numpy_iterator():
    state, loss = train_step(state, batch['image'], batch['label'])
