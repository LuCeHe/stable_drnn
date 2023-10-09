import time
from functools import partial
from tqdm import tqdm

import jax
import jax.ops
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from flax import linen as nn
import optax

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.minimal_LRU_modified.lru.model import LRU

batch_size = 8
time_steps = 7
features = 16

ssm = partial(LRU, d_hidden=features, d_model=features)

BatchClassificationModel = nn.vmap(
    SequenceLayer,
    in_axes=0, out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')

model = partial(
    BatchClassificationModel, ssm=ssm, dropout=.1, d_model=features, activation='full_glu',
)(training=True)

print(model)

inps = jnp.arange(0, time_steps)
inps = jnp.tile(inps, (batch_size, 1))
inps = jnp.repeat(inps[:, :, jnp.newaxis], features, axis=2)
inps = jnp.float32(inps)

jax_seed = 0
key = random.PRNGKey(jax_seed)
init_rng, pretrain_rng, wshuff_rng = random.split(key, num=3)
dummy_input = jnp.ones((batch_size, time_steps, features))
init_rng, dropout_rng = jax.random.split(init_rng, num=2)

variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

params = variables["params"]


def compute_radius(i, Jb_osb, Jb):
    j_t = Jb_osb[:, i, :, i]
    radius_t = jnp.linalg.norm(j_t, axis=(1, 2))

    j_l = Jb[:, i, :, i]
    radius_l = jnp.linalg.norm(j_l, axis=(1, 2))
    return radius_t, radius_l


def get_radiuses(params, dropout_rng, input_batch):
    f = lambda x: model.apply({'params': params}, x, rngs={'dropout': dropout_rng})

    # calculate the jacobian
    Jb = jax.jacfwd(f)(input_batch)

    # remove cross batch elements
    Jb = jnp.diagonal(Jb, axis1=0, axis2=3)
    Jb = jnp.moveaxis(Jb, [-1, ], [0, ])

    # remove the first time step
    Jb_back = Jb[:, 1:]

    # compute the radiuses in the time and layer dimensions
    radiuses_t, radiuses_l = jax.vmap(lambda i: compute_radius(i, Jb_back, Jb))(jnp.arange(time_steps - 1))
    return radiuses_t, radiuses_l


target_norm = 1
tnt, tnl = target_norm, target_norm


# TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, inputs, do_rng, wshuff=True):
    def loss_fn(params):
        if wshuff:
            print('Shuffling weights during pretraining!')
            for k, v in params.items():
                for sk, sv in v.items():
                    params[k][sk] = random.shuffle(wshuff_rng, sv)
        rt, rl = state.apply_fn(params, do_rng, inputs)
        loss = jnp.mean((rt - tnt) ** 2) + jnp.mean((rl - tnl) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


tx = optax.adabelief(learning_rate=0.05)

state = TrainState.create(
    apply_fn=get_radiuses,
    params=variables['params'],
    tx=tx,
)

pretrain_steps = 2  # 1000
# print(variables['params'])

for _ in tqdm(range(pretrain_steps)):
    # inputs as random samples of shape (batch_size, time_steps, features)
    inputs = random.normal(pretrain_rng, (batch_size, time_steps, features))
    state, loss = train_step(state, inputs, dropout_rng)
    if loss < 1e-3:
        print('Early stopping on loss < 1e-3: ', loss)
        break
    # print(loss)
# print(state.params)

print(variables['params'].keys())
# compare params stats before and after, pretraining
for (ko, vo), (kn, vn) in zip(variables['params'].items(), state.params.items()):
    print('-' * 20)
    print(ko, kn)
    # print(vo)
    for sko, skn in zip(vo.keys(), vn.keys()):
        print(sko, skn)
        print(f'mean/std old { jnp.mean(vo[sko])}/{jnp.std(vo[sko])}')
        print(f'mean/std new { jnp.mean(vn[skn])}/{jnp.std(vn[skn])}')
