import time
from functools import partial

from tqdm.auto import tqdm
from pyaromatics.stay_organized.utils import str2val

import jax
import jax.ops
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jax import random
from flax import linen as nn
import optax


def compute_radius(i, Jb_osb, Jb):
    j_t = Jb_osb[:, i, :, i]
    Sigma_t = jnp.linalg.svd(j_t, compute_uv=False)
    eigs_t = jnp.sqrt(Sigma_t)
    radius_t = jnp.max(eigs_t, axis=(-1,))

    j_l = Jb[:, i, :, i]
    Sigma_l = jnp.linalg.svd(j_l, compute_uv=False)
    eigs_l = jnp.sqrt(Sigma_l)
    radius_l = jnp.max(eigs_l, axis=(-1,))

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
        return loss, [jnp.mean(rtm), jnp.mean(rlm), jnp.std(rtm), jnp.std(rlm)]

    (loss, norms), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, norms


def pretrain(
        model, jax_seed, batch_size, time_steps, features, comments='', ptcomments='', pretrain_steps=3000, plot=False,
        loss_threshold=3.7, ptlr=0.05, optimizer='adam'):
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
    elif optimizer == 'adamw':
        tx = optax.adamw(learning_rate=ptlr)
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

    multiply = True
    shuffling = True
    mult_period = 50
    shuff_period = 50
    optch_period = 400
    opt_changes = 0

    ptlosses = []
    tnorms = []
    lnorms = []
    tnorms_std = []
    lnorms_std = []

    with tqdm(total=pretrain_steps) as pbar:
        for step in range(1, pretrain_steps + 1):
            # inputs as random samples of shape (batch_size, time_steps, features)
            inputs = random.normal(pretrain_rng, (batch_size, time_steps, features))
            # uniform samples
            # inputs = random.uniform(pretrain_rng, (batch_size, time_steps, features), minval=-jnp.sqrt(3),
            #                         maxval=jnp.sqrt(3))
            state, loss, norms = train_step(state, inputs, dropout_rng, tnt, tnl, wshuff_rng)
            nt, nl = norms[0], norms[1]
            tnorms.append(float(norms[0]))
            lnorms.append(float(norms[1]))
            tnorms_std.append(float(norms[2]))
            lnorms_std.append(float(norms[3]))

            pbar.set_description(
                f"Pre-training Loss: {loss:.4f}, nt: {norms[0]:.2f}\u00B1{norms[2]:.2f}, nl: {norms[1]:.2f}\u00B1{norms[3]:.2f}",
                refresh=True)
            pbar.update(1)

            if 'changeopt' in ptcomments and step % optch_period == 0:
                multiply = True
                opt_changes += 1
                if opt_changes % 2 == 1:
                    lr = ptlr * .5
                    # tx2 = optax.sgd(learning_rate=lr, momentum=0.7)
                    tx2 = optax.adabelief(learning_rate=lr)

                    shuff_period = 20
                    optch_period = 200
                    shuffling = False
                    print(f'AdaBelief lr={lr}')
                else:
                    lr = ptlr * .1
                    tx2 = optax.adamw(learning_rate=lr)
                    shuff_period = 20
                    optch_period = 200
                    shuffling = False
                    print(f'AdamW lr={lr}')

                tx2 = optax.chain(
                    tx2,
                    optax.zero_nans(),
                    optax.clip_by_global_norm(1.0),
                    optax.ema(0.3),
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

            if 'wmultiplier' in ptcomments and multiply and step % mult_period == 0:
                print('Multiplying weights')

                for k, v in state.params.items():
                    for sk, sv in v.items():
                        print(k, sk)
                        m = 1

                        time_multiplier = False
                        depth_multiplier = False

                        # if 'nu_log' in sk:
                        #     time_multiplier = True

                        # elif 'Lambda_re' in sk or 'Lambda_im' in sk:
                        #     time_multiplier = True

                        if 'C_re' == sk or 'C_im' == sk or 'B_re' == sk or 'B_im' == sk:
                            depth_multiplier = True

                        elif 'C1' == sk or 'C2' == sk or 'B' == sk or 'D' == sk:
                            depth_multiplier = True

                        elif sk == 'bias' or sk == 'scale':
                            depth_multiplier = True
                            print(sk, jnp.mean(sv), jnp.std(sv))

                        # elif 'kernel' in sk:
                        #     depth_multiplier = True

                        if time_multiplier:
                            m = tnt / nt

                        if depth_multiplier:
                            m = tnl / nl

                        if sk == 'bias' or sk == 'scale':
                            m = 1 / m

                        m = jnp.clip(m, .9, 1.1)
                        print('multiplier:', m)
                        state.params[k][sk] = m * sv
                state = state.replace(params=state.params)

            ptlosses.append(float(loss))
            if loss < loss_threshold:
                print(f'Early stopping on loss < {loss_threshold}: {loss}')
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

    results = {
        'pretraining_loss': ptlosses, 'tnorms': tnorms, 'lnorms': lnorms,
        'tnorms_std': tnorms_std, 'lnorms_std': lnorms_std
    }
    return new_params, results


if __name__ == '__main__':
    from alif_sg.S5.s5.layers import SequenceLayer

    batch_size = 8
    time_steps = 7
    features = 128
    model_name = 'lru'  # lru s5

    if model_name == 'lru':
        from alif_sg.S5.s5.lru_model import LRU

        d_hidden = int(features * 1.5)

        ssm = partial(LRU, d_hidden=d_hidden, d_model=features)
    elif model_name == 's5':
        from alif_sg.S5.s5.ssm import init_S5SSM
        from alif_sg.S5.s5.ssm_init import make_DPLR_HiPPO
        from jax._src.scipy.linalg import block_diag

        blocks = 4
        # determine the size of initial blocks
        ssm_size = int(features * 1.5 // 2)
        block_size = int(ssm_size / blocks)

        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        ssm = init_S5SSM(
            H=features,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init='lecun_normal',
            discretization='zoh',
            dt_min=1,
            dt_max=2,
            conj_sym=False,
            clip_eigs=False,
            bidirectional=True
        )

    BatchClassificationModel = nn.vmap(
        SequenceLayer,
        in_axes=0, out_axes=0,
        variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
        split_rngs={"params": False, "dropout": True}, axis_name='batch')

    model = partial(
        BatchClassificationModel, ssm=ssm, dropout=.1, d_model=features, activation='full_glu',
    )(training=True)

    print(model)

    new_params, presults = pretrain(
        model, 0, batch_size=batch_size, pretrain_steps=10,
        time_steps=time_steps, features=features, loss_threshold=0.1,
        # ptcomments='wmultiplier'
    )
