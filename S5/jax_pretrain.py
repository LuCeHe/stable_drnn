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
            )[0]
        else:
            f = lambda x: model.apply(
                {"params": params}, x, rngs={"dropout": dropout_rng},
                mutable=["intermediates"],
            )[0]

        # calculate the jacobian
        Jb = jax.jacfwd(f)(input_batch)

        # remove cross batch elements
        Jb = jnp.diagonal(Jb, axis1=0, axis2=3)
        Jb = jnp.moveaxis(Jb, [-1, ], [0, ])

        # remove the first time step
        Jb_back = Jb[:, 1:]

        # compute the radiuses in the time and layer dimensions
        time_steps = Jb_back.shape[1]
        radiuses_t, radiuses_l = jax.vmap(lambda i: compute_radius(i, Jb_back, Jb))(jnp.arange(time_steps - 1))

        return radiuses_t, radiuses_l

    return _get_radiuses


@jax.jit
def train_step(state, inputs, do_rng, tnt, tnl, wshuff_rng):
    def loss_fn(params):
        rt, rl = state.apply_fn(params, state, do_rng, inputs)

        rtm = jnp.mean(rt, axis=(1,))
        rlm = jnp.mean(rl, axis=(1,))

        loss = jnp.mean(jnp.abs(rtm - tnt)) + jnp.mean(jnp.abs(rlm - tnl))
        # loss = jnp.mean((rtm - tnt)**2) + jnp.mean((rlm - tnl)**2)
        # loss = jnp.sqrt(jnp.mean((rtm - tnt)**2) + jnp.mean((rlm - tnl)**2))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


def pretrain(
        model, jax_seed, batch_size, time_steps, features, comments='', ptcomments='', pretrain_steps=3000, plot=False,
        loss_threshold=1e-3, ptlr=0.05, optimizer='adam'):
    target_norm = str2val(comments, 'targetnorm', float, default=1)

    tnt, tnl = target_norm, target_norm
    if 'unbalanced' in comments:
        tnl = 0.1
        tnt = .9

    print(f'Targets: tl={tnl}, tt={tnt}')

    key = random.PRNGKey(jax_seed)
    init_rng, pretrain_rng, wshuff_rng = random.split(key, num=3)
    dummy_input = jnp.ones((batch_size, time_steps, features))
    init_rng, dropout_rng = jax.random.split(init_rng, num=2)

    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)

    if optimizer == 'adam':
        tx = optax.adam(learning_rate=ptlr)
    elif optimizer == 'adabelief':
        tx = optax.adabelief(learning_rate=ptlr)
    elif optimizer == 'rmsprop':
        tx = optax.rmsprop(learning_rate=ptlr)
    elif optimizer == 'sgd':
        tx = optax.sgd(learning_rate=ptlr, momentum=0.3)
    elif optimizer == 'nsgd':
        tx = optax.noisy_sgd(learning_rate=ptlr)
    elif optimizer == 'lion':
        tx = optax.lion(learning_rate=ptlr)
    else:
        tx = optax.sgd(learning_rate=ptlr)

    if 'nonan' in ptcomments:
        tx = optax.chain(
            tx,
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
            optax.ema(0.9),
            # optax.add_decayed_weights(weight_decay=0.01),
        )

    aux_dict = {}
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

    lr = 0.1
    shuffling = True
    shuff_period = 50
    opt_changes = 0
    with tqdm(total=pretrain_steps) as pbar:
        for step in range(1, pretrain_steps + 1):
            # inputs as random samples of shape (batch_size, time_steps, features)
            # inputs = random.normal(pretrain_rng, (batch_size, time_steps, features))
            # uniform samples
            inputs = random.uniform(pretrain_rng, (batch_size, time_steps, features), minval=-jnp.sqrt(3),
                                    maxval=jnp.sqrt(3))
            state, loss = train_step(state, inputs, dropout_rng, tnt, tnl, wshuff_rng)

            pbar.set_description(f"Pre-training Loss: {loss:.4f}", refresh=True)
            pbar.update(1)

            if 'changeopt' in ptcomments and step % 300 == 0:
                opt_changes += 1
                lr = lr * .3
                if opt_changes % 2 == 1:
                    print('Adam')
                    lr = 0.01
                    # tx2 = optax.sgd(learning_rate=lr, momentum=0.7)
                    # tx2 = optax.adamw(learning_rate=lr, weight_decay=0.01)
                    tx2 = optax.adabelief(learning_rate=lr)
                    # tx2 = optax.optimistic_gradient_descent(learning_rate=lr)
                    shuff_period = 300
                else:
                    print('SGD')
                    lr = 1
                    tx2 = optax.sgd(learning_rate=lr, momentum=0.7)
                    shuff_period = 100

                tx2 = optax.chain(
                    tx2,
                    optax.zero_nans(),
                    optax.clip_by_global_norm(1.0),
                    optax.ema(0.8),
                    optax.add_decayed_weights(weight_decay=0.01),
                )

                opt_state = tx2.init(state.params)
                state = state.replace(tx=tx2)
                state = state.replace(opt_state=opt_state)

                # shuffling = False
                print('Changing optimizer')

            if 'wshuffle' in ptcomments and step % shuff_period == 0 and shuffling:
                wshuff_rng, new_wshuff_rng = random.split(wshuff_rng)
                for k, v in state.params.items():
                    for sk, sv in v.items():
                        state.params[k][sk] = random.shuffle(new_wshuff_rng, sv)
                state = state.replace(params=state.params)

                print('Shuffling weights')

            if loss < loss_threshold:
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
