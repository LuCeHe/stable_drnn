import time
from functools import partial
from tqdm.auto import tqdm

import jax
import jax.ops
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from flax import linen as nn
import optax

from alif_sg.S5.s5.layers import SequenceLayer
from alif_sg.minimal_LRU_modified.lru.model import LRU
from pyaromatics.stay_organized.utils import str2val


def compute_radius(i, Jb_osb, Jb):
    j_t = Jb_osb[:, i, :, i]
    radius_t = jnp.linalg.norm(j_t, axis=(1, 2))

    j_l = Jb[:, i, :, i]
    radius_l = jnp.linalg.norm(j_l, axis=(1, 2))
    return radius_t, radius_l


def get_radiuses(model, aux_dict):
    batchnorm = False
    if 'batch_stats' in aux_dict.keys():
        batchnorm = True

    def _get_radiuses(params, state, dropout_rng, input_batch):

        if batchnorm:
            f = lambda x: model.apply(
                {"params": params, "batch_stats": state.batch_stats}, x, rngs={"dropout": dropout_rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            f = lambda x: model.apply(
                {"params": params}, x, rngs={"dropout": dropout_rng},
                mutable=["intermediates"],
            )

        # calculate the jacobian
        Jb = jax.jacfwd(f)(input_batch)
        print(Jb[0].shape, Jb[1].shape)
        print('Jb shape: ', Jb.shape)

        # remove cross batch elements
        Jb = jnp.diagonal(Jb, axis1=0, axis2=3)
        Jb = jnp.moveaxis(Jb, [-1, ], [0, ])

        # remove the first time step
        Jb_back = Jb[:, 1:]

        # compute the radiuses in the time and layer dimensions
        radiuses_t, radiuses_l = jax.vmap(lambda i: compute_radius(i, Jb_back, Jb))(jnp.arange(time_steps - 1))
        return radiuses_t, radiuses_l

    return _get_radiuses


@jax.jit
def train_step(state, inputs, do_rng, tnt, tnl, wshuff_rng):
    def loss_fn(params):
        # shuffling weights during pretraining
        for k, v in params.items():
            for sk, sv in v.items():
                params[k][sk] = random.shuffle(wshuff_rng, sv)
        rt, rl = state.apply_fn(params, state, do_rng, inputs)
        loss = jnp.mean((rt - tnt) ** 2) + jnp.mean((rl - tnl) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def pretrain(model, jax_seed, batch_size, time_steps, features, comments='', pretrain_steps=3000, plot=False):
    target_norm = str2val(comments, 'targetnorm', float, default=1)
    tnt, tnl = target_norm, target_norm
    if 'unbalanced' in comments:
        tnl = 0.1
        tnt = .9

    key = random.PRNGKey(jax_seed)
    init_rng, pretrain_rng, wshuff_rng = random.split(key, num=3)
    dummy_input = jnp.ones((batch_size, time_steps, features))
    init_rng, dropout_rng = jax.random.split(init_rng, num=2)

    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

    tx = optax.adabelief(learning_rate=0.05)

    aux_dict = {}
    print('variables keys: ', variables.keys())
    TS = TrainState
    if 'batch_stats' in variables.keys():
        from typing import Any

        aux_dict = {'batch_stats': variables["batch_stats"]}

        class TS(TrainState):
            batch_stats: Any

    state = TS.create(
        apply_fn=get_radiuses(model, aux_dict),
        params=variables['params'],
        tx=tx, **aux_dict
    )

    with tqdm(total=pretrain_steps) as pbar:
        for _ in range(pretrain_steps):
            # inputs as random samples of shape (batch_size, time_steps, features)
            inputs = random.normal(pretrain_rng, (batch_size, time_steps, features))
            state, loss = train_step(state, inputs, dropout_rng, tnt, tnl, wshuff_rng)

            pbar.set_description(f"Loss: {loss:.4f}", refresh=True)
            pbar.update(1)
            if loss < 1e-3:
                print('Early stopping on loss < 1e-3: ', loss)
                break

    if plot:
        # plot the weights before and after pretraining
        print(variables['params'].keys())
        n_weights = sum([1 for k, v in variables['params'].items() for sk, sv in v.items()])
        print(f'Number of weights tensors: {n_weights}')

        # subplots
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axs = plt.subplots(int(np.sqrt(n_weights)) + 1, int(np.sqrt(n_weights)) + 1, figsize=(10, 5))

        i, j = 0, 0

        # compare params stats before and after, pretraining
        for (ko, vo), (kn, vn) in zip(variables['params'].items(), state.params.items()):
            for sko, skn in zip(vo.keys(), vn.keys()):
                axs[i, j].hist(vo[sko].flatten(), bins=100, alpha=.5, label='old', color='r')
                axs[i, j].hist(vn[skn].flatten(), bins=100, alpha=.5, label='new', color='b')

                axs[i, j].set_title(f'{ko}_{sko}')

                j += 1
                if j == int(np.sqrt(n_weights)) + 1:
                    j = 0
                    i += 1

        plt.show()

    new_params = state.params
    return new_params, loss


if __name__ == '__main__':
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

    jax_seed = 0
    key = random.PRNGKey(jax_seed)
    init_rng, pretrain_rng, wshuff_rng = random.split(key, num=3)
    dummy_input = jnp.ones((batch_size, time_steps, features))
    init_rng, dropout_rng = jax.random.split(init_rng, num=2)

    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

    # params = variables["params"]

    new_params, pretraining_loss = pretrain(model, variables)
