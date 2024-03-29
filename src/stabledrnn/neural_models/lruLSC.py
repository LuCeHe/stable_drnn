import os, random
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm


try:
    from stable_drnn.neural_models.normify import get_norms
except:
    import sys

    sys.path.append('..')
    from stabledrnn.neural_models.normify import get_norms

from innocent_explorations.lsc_unused.admin_model_removal import get_pretrained_file
from lruun.tf.linear_recurrent_unit import ResLRUCell, ResLRUFFN
from pyaromatics.stay_organized.utils import str2val

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
GEXPERIMENTS = os.path.abspath(os.path.join(CDIR, '..', 'good_experiments'))
os.makedirs(GEXPERIMENTS, exist_ok=True)


def load_resLSC_model(path):
    model = tf.keras.models.load_model(
        path,
        custom_objects={
            'ResLRUCell': ResLRUCell, 'ResLRUFFN': ResLRUFFN
        },
        compile=False
    )
    return model


def lruLSC(
        comments='findLSC_radius', seed=0, stack=4, width=32, classes=2, vocab_size=7, maxlen=1,
        batch_size=8
):
    net_name = 'reslru'
    task_name = 'anytask'

    pretrained_file = get_pretrained_file(comments, seed, net_name, task_name, stack)
    path_pretrained = os.path.join(GEXPERIMENTS, pretrained_file)
    ffn_stem = None
    if os.path.exists(path_pretrained):
        print('Loading pretrained lsc weights')
        ffn_stem = load_resLSC_model(path_pretrained)

    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    time_steps = 1

    n_layers = int(stack)
    ts = 50 if 'test' in comments else 500  # number of pretraining steps
    tc = n_layers * 2  # time constant for the moving averages
    round_to = 5
    decay = .97
    rand = lambda shape=(width,): tf.random.normal(shape)

    target_norm = str2val(comments, 'targetnorm', float, default=1)
    tn_l = target_norm
    if target_norm == 0.5 and 'unbalanced' in comments:
        tn_l = n_layers / (n_layers + maxlen)

    tn_t = target_norm
    if target_norm == 0.5 and 'unbalanced' in comments:
        tn_t = maxlen / (n_layers + maxlen)

    if 'onlyloadpretrained' in comments:
        ts = 10
    else:
        comments += 'wmultiplier_wshuff'

    cells = [ResLRUCell(num_neurons=width) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

    if not ffn_stem is None:
        rnns_nrs = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in cells]
        rnn_stem = tf.keras.models.Sequential(rnns_nrs)
        rnn_stem.build((None, None, width))
        rnn_stem.set_weights(ffn_stem.get_weights())
        del rnn_stem

    pbar = tqdm(total=ts)
    li = None
    ni = None
    ali = None
    ati = None
    sti = None

    ma_loss = None
    ma_norm = None
    std_ma_norm = None
    for t in tqdm(range(ts)):
        try:

            inputs_t1 = tf.Variable(rand((batch_size, time_steps, width)))
            init_states_l1_t1 = [
                tf.Variable(rand((batch_size, 2 * width))),
                tf.Variable(rand((batch_size, 2 * width)))
            ]
            init_states_l2_t1 = [
                tf.Variable(rand((batch_size, 2 * width))),
                tf.Variable(rand((batch_size, 2 * width)))
            ]

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs_t1)
                tape.watch(init_states_l1_t1)
                tape.watch(init_states_l2_t1)

                # pick 2 rnns randomly
                np.random.shuffle(rnns)
                rnn_l1 = rnns[0]
                rnn_l2 = rnns[1]

                # concatenate states
                states_l1_conc = tf.concat(init_states_l1_t1, axis=-1)
                # deconcatenate states
                states_l1_deconc = tf.split(states_l1_conc, 2, axis=-1)
                all_outs_l1_t1 = rnn_l1(inputs_t1, states_l1_deconc)
                output_l1_t2, states_l1_t2 = all_outs_l1_t1[0], all_outs_l1_t1[1:]

                # concatenate states
                states_l2_conc = tf.concat(states_l1_t2, axis=-1)

                all_outs_l2_t1 = rnn_l2(output_l1_t2, init_states_l2_t1)
                output_l2_t2, states_l2_t2 = all_outs_l2_t1[0], all_outs_l2_t1[1:]

                # concatenate states
                states_l3_conc = tf.concat(states_l2_t2, axis=-1)

            # M_t jacobian of states_l2_conc w.r.t. states_l1_conc
            a_t, loss_t, _ = get_norms(
                tape=tape, lower_states=[states_l1_conc], upper_states=[states_l2_conc], comments=comments,
                target_norm=tn_t
            )

            # M_l jacobian of states_l3_conc w.r.t. states_l1_conc

            a_l, loss_l, _ = get_norms(
                tape=tape, lower_states=[states_l1_conc], upper_states=[states_l3_conc], comments=comments,
                target_norm=tn_l
            )

            mean_loss = tf.reduce_mean(loss_t + loss_l).numpy().astype(np.float32)
            mean_norm = tf.reduce_mean((a_t + a_l) / 2).numpy().astype(np.float32)

            ma_loss = mean_loss if ma_loss is None else ma_loss * (tc - 1) / tc + mean_loss / tc
            ma_norm = mean_norm if ma_norm is None else ma_norm * (tc - 1) / tc + mean_norm / tc
            current_std = np.std([*a_t.numpy().tolist(), *a_l.numpy().tolist()])
            std_ma_norm = current_std if std_ma_norm is None else std_ma_norm * (tc - 1) / tc + np.std(current_std) / tc

            if li == None:
                li = str(mean_loss.round(round_to))

            if ni == None:
                ni = str(mean_norm.round(round_to))

            if ali == None:
                ali = str(np.mean(a_l.numpy()).round(round_to))

            if ati == None:
                ati = str(np.mean(a_t.numpy()).round(round_to))

            if sti == None:
                sti = str(np.std([*a_t.numpy().tolist(), *a_l.numpy().tolist()]).round(round_to))

            wnames_1 = [weight.name for weight in rnn_l1.weights]
            wnames_2 = [weight.name for weight in rnn_l2.weights]

            weights_1 = rnn_l1.trainable_weights
            weights_2 = rnn_l2.trainable_weights

            if 'wmultiplier' in comments:
                for pair, (weights, wnames, rnn) in enumerate(zip(
                        [weights_1, weights_2],
                        [wnames_1, wnames_2],
                        [rnn_l1, rnn_l2])
                ):
                    new_weights = []

                    for w, wname in zip(weights, wnames):
                        multiplier = 1

                        depth_radius = False
                        if ('C_re' in wname or 'B_re' in wname or 'B_im' in wname or 'C_im' in wname) and pair == 1:
                            depth_radius = True

                        rec_radius = False
                        if 'lambda_nu' in wname:
                            rec_radius = True

                        if depth_radius:
                            local_norm = a_l
                            multiplier = tn_l / local_norm

                        elif rec_radius:
                            local_norm = a_t
                            multiplier = tn_t / local_norm

                        m = tf.reduce_mean(multiplier).numpy()
                        m = np.clip(m, 0.95, 1.05)

                        w = m * w.numpy()

                        if 'wshuff' in comments:
                            oshape = w.shape
                            w = w.reshape(-1)
                            np.random.shuffle(w)
                            w = w.reshape(oshape)

                        new_weights.append(w)

                    rnn.set_weights(new_weights)

                w1 = rnn_l1.get_weights()
                w2 = rnn_l2.get_weights()

                # mix half of the weights
                for i in range(len(w1)):
                    w1i = w1[i]
                    w2i = w2[i]
                    oshape = w1i.shape
                    w1i = w1i.reshape(-1)
                    w2i = w2i.reshape(-1)
                    # move half the weights to the other rnn
                    w1i[:len(w1i) // 2], w2i[:len(w2i) // 2] = w2i[:len(w2i) // 2], w1i[:len(w1i) // 2]
                    w1i = w1i.reshape(oshape)
                    w2i = w2i.reshape(oshape)
                    w1[i] = w1i
                    w2[i] = w2i

                rnn_l1.set_weights(w1)
                rnn_l2.set_weights(w2)

            pbar.set_description(
                f"Step {t}; "
                f"loss {str(np.array(ma_loss).round(round_to))}/{li}; "
                f"mean norms {np.array(ma_norm).round(round_to)}/{ni}; "
                f"ma std norms {str(np.array(std_ma_norm).round(round_to))}/{sti}; "
                f"at {str(np.mean(a_t.numpy()).round(round_to))}/{ati} ({str(np.array(tn_t).round(round_to))}); "
                f"al {str(np.mean(a_l.numpy()).round(round_to))}/{ali} ({str(np.array(tn_l).round(round_to))}); "
            )
        except Exception as e:
            print(e)

    results = {
        'ma_loss': ma_loss,
        'ma_norm': ma_norm,
        'std_ma_norm': std_ma_norm,
        'a_t': np.mean(a_t.numpy()),
        'a_l': np.mean(a_l.numpy()),
        'loss_t': np.mean(loss_t.numpy()),
        'loss_l': np.mean(loss_l.numpy()),
        'li': li,
        'ni': ni,
        'ali': ali,
        'ati': ati,
        'sti': sti,
        'target_norm': target_norm,
        'tn_t': tn_t,
        'tn_l': tn_l,
        'lsc_comments': comments,
    }

    scales = compare_to_default_scales(width, n_layers, cells)
    results.update(scales)

    ffn_weights = equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=cells,
                                       path_pretrained=path_pretrained)

    results['final_norm_dec'] = None
    results['final_norms_mean'] = mean_norm
    results['final_norms_std'] = current_std
    results['std_ma_norm'] = std_ma_norm
    results['best_std_ma_norm'] = std_ma_norm

    del tape, rnn_l1, rnn_l2, inputs_t1, init_states_l1_t1, init_states_l2_t1, states_l1_conc, states_l1_deconc, \
        all_outs_l1_t1, output_l1_t2, states_l1_t2, states_l2_conc, all_outs_l2_t1, output_l2_t2, states_l2_t2, \
        a_t, loss_t, a_l, loss_l, mean_loss, mean_norm, ma_loss, ma_norm, current_std, std_ma_norm, wnames_1, \
        wnames_2, weights_1, weights_2, new_weights, w1, w2, oshape, w1i, w2i, i, pair, multiplier, depth_radius, \
        local_norm, m, w, wname, rec_radius, rnn, t, ts, tc, round_to, decay, rand, target_norm, tn_l, tn_t, \
        pretrained_file, path_pretrained, ffn_stem, comments, seed, stack, width, classes, vocab_size, maxlen, \
        batch_size, li, ni, ali, ati, sti, cells, rnns, pbar, time_steps, n_layers, \
        wnames, weights

    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()

    return ffn_weights, results


def equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=None, path_pretrained=None,
                         rec_weights=None):
    if cells is None:
        cells = [ResLRUCell(num_neurons=width) for _ in range(n_layers)]

    rnns = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in cells]
    rnn_stem = tf.keras.models.Sequential(rnns)

    if not rec_weights is None:
        rnn_stem = tf.keras.models.Sequential(rnns)
        rnn_stem.build((None, None, width))
        rnn_stem.set_weights(rec_weights)

    emb = tf.keras.layers.Embedding(vocab_size, width)
    dense = tf.keras.layers.Dense(classes)
    rnn_model = tf.keras.models.Sequential(
        [emb, rnn_stem] + ([dense] if not 'embproj' in comments else [])
    )

    ffns = [ResLRUFFN(num_neurons=width) for _ in range(n_layers)]
    ffn_stem = tf.keras.models.Sequential(ffns)

    ffn_model = tf.keras.models.Sequential(
        [emb, ffn_stem] + ([dense] if not 'embproj' in comments else [])
    )
    ffn_model.set_weights(rnn_model.get_weights())

    # save stem ffn
    if path_pretrained is not None:
        print('Saving pretrained lsc weights')
        print(path_pretrained)
        for i in range(len(ffn_stem.weights)):
            ffn_stem.weights[i]._handle_name = ffn_stem.weights[i].name + "_" + str(i)
        ffn_stem.save(path_pretrained)

    return ffn_model.get_weights()


def compare_to_default_scales(width, n_layers, pretrained_cells):
    new_cells = [ResLRUCell(num_neurons=width) for _ in range(n_layers)]
    new_rnns = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in new_cells]
    new_rnn_stem = tf.keras.models.Sequential(new_rnns)

    pretrained_rnns = [tf.keras.layers.RNN(pretrained_cells, return_sequences=True) for cell in new_cells]
    pretrained_rnn_stem = tf.keras.models.Sequential(pretrained_rnns)

    new_rnn_stem.build((None, None, width))
    pretrained_rnn_stem.build((None, None, width))

    weight_names = [weight.name for weight in pretrained_rnn_stem.weights]
    pretrained_weights = pretrained_rnn_stem.get_weights()
    new_weights = new_rnn_stem.get_weights()
    scales = {}
    for wn, pw, nw in zip(weight_names, pretrained_weights, new_weights):

        if not np.std(nw) == 0:
            scale = np.std(pw) / np.std(nw)
        elif np.std(pw) == np.std(nw):
            scale = 1
        else:
            scale = np.inf

        scales[wn + '_scale'] = scale
        print(wn, scale)
    return scales


def lruLSCffn(
        comments='findLSC_radius_test', seed=0, stack=4, width=3, classes=2, vocab_size=7, maxlen=4,
        batch_size=8, loss_threshold=0.01, max_steps=1000
):
    net_name = 'reslruffn'
    task_name = 'anytask'

    pretrained_file = get_pretrained_file(comments, seed, net_name, task_name, stack)
    path_pretrained = os.path.join(GEXPERIMENTS, pretrained_file)

    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)

    time_steps = maxlen

    n_layers = int(stack)
    ts = 2 if 'test' in comments else max_steps  # number of pretraining steps
    tc = n_layers * 2  # time constant for the moving averages
    round_to = 5
    rand = lambda shape=(width,): tf.random.normal(shape)

    target_norm = str2val(comments, 'targetnorm', float, default=1)
    tn_l = target_norm
    if target_norm == 0.5 and 'unbalanced' in comments:
        tn_l = n_layers / (n_layers + maxlen)

    tn_t = target_norm
    if target_norm == 0.5 and 'unbalanced' in comments:
        tn_t = maxlen / (n_layers + maxlen)

    if 'onlyloadpretrained' in comments:
        ts = 10
    else:
        comments += '_wshuff'
        # comments += '_wmultiplier'
        pass

    optimizer = None
    if not 'nosgd' in comments:
        adabelief = tfa.optimizers.AdaBelief(lr=.01, weight_decay=0.004)
        optimizer = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

    ffns = [ResLRUFFN(num_neurons=width) for _ in range(n_layers)]

    inputs = tf.Variable(rand((batch_size, time_steps, width)))
    for ffn in ffns:
        out = ffn(inputs)
        del out
    del inputs
    wnames = [weight.name for weight in ffn.weights]
    print(wnames)

    pbar = tqdm(total=ts)
    li = None
    ni = None
    ali = None
    ati = None
    sti = None

    ma_loss = None
    ma_norm = None
    std_ma_norm = None
    for t in tqdm(range(ts)):

        try:
            with tf.GradientTape(persistent=True) as tape:

                # ffns = [ResLRUFFN(num_neurons=width) for _ in range(n_layers)]
                # ffn = tf.keras.models.Sequential(ffns)
                random.shuffle(ffns)
                ffn = ffns[0]
                inputs = tf.Variable(rand((batch_size, time_steps, width)))
                # tape.watch(inputs)
                out = ffn(inputs)

                # compute radiuses
                hs = tape.batch_jacobian(out, inputs, experimental_use_pfor=True)

                # reorder axis to be (batch, time, time, width, width)
                hs = tf.transpose(hs, perm=[0, 1, 3, 2, 4])
                eigs, _ = tf.linalg.eig(hs)

                radius = tf.reduce_max(tf.abs(eigs), axis=[-1])
                radius_1 = radius[:, 1:, :-1]  # this is the good one
                # radius_1 = radius[:, 2:, :-2]

                # diagonal of the radius_1
                rt = tf.linalg.diag_part(radius_1)
                rl = tf.linalg.diag_part(radius)

                loss_t = tf.reduce_mean(tf.square(tf.reduce_mean(rt, axis=0) - tn_t))
                loss_l = tf.reduce_mean(tf.square(tf.reduce_mean(rl, axis=0) - tn_l))

                mean_loss = (loss_t + loss_l) / 2

            if not 'nosgd' in comments:
                grads = tape.gradient(mean_loss, ffn.trainable_weights)
                optimizer.apply_gradients(zip(grads, ffn.trainable_weights))
                del grads

            if 'wmultiplier' in comments:
                # print('-' * 100)
                new_weights = []
                weights = ffn.get_weights()

                for w, wname in zip(weights, wnames):
                    # print(wname)
                    multiplier = 1

                    depth_radius = False
                    if 'C_re' in wname or 'B_re' in wname or 'B_im' in wname or 'C_im' in wname:
                        depth_radius = True
                    # if 'kernel' in wname:
                    #     depth_radius = True

                    rec_radius = False
                    if 'lambda_nu' in wname:
                        rec_radius = True

                    if depth_radius:
                        local_norm = rl
                        multiplier = tn_l / local_norm
                        print(wname, tf.reduce_mean(multiplier).numpy())

                    elif rec_radius:
                        local_norm = rt
                        multiplier = tn_t / local_norm
                        # multiplier = 1/multiplier
                        print(wname, tf.reduce_mean(multiplier).numpy())

                    m = tf.reduce_mean(multiplier).numpy()
                    m = np.clip(m, 0.85, 1.15)

                    w = m * w

                    if 'wshuff' in comments:
                        oshape = w.shape
                        w = w.reshape(-1)
                        np.random.shuffle(w)
                        w = w.reshape(oshape)

                    new_weights.append(w)

                ffn.set_weights(new_weights)

            mean_norm = ((tf.reduce_mean(rt) + tf.reduce_mean(rl)) / 2).numpy().astype(np.float32)
            ml = mean_loss.numpy().astype(np.float32)
            ma_loss = ml if ma_loss is None else ma_loss * (tc - 1) / tc + ml / tc
            ma_norm = mean_norm if ma_norm is None else ma_norm * (tc - 1) / tc + mean_norm / tc

            current_std = (np.std(rt) + np.std(rl)) / 2
            std_ma_norm = current_std if std_ma_norm is None else std_ma_norm * (tc - 1) / tc + np.std(current_std) / tc

            if li == None:
                li = str(ml.round(round_to))
                ni = str(mean_norm.round(round_to))
                ali = str(np.mean(rl.numpy()).round(round_to))
                ati = str(np.mean(rt.numpy()).round(round_to))
                sti = str(current_std.round(round_to))

            pbar.set_description(
                f"Step {t}; "
                f"loss {str(np.array(ma_loss).round(round_to))}/{li}; "
                f"mean norms {np.array(ma_norm).round(round_to)}/{ni}; "
                f"ma std norms {str(np.array(std_ma_norm).round(round_to))}/{sti}; "
                f"at {str(np.mean(rt.numpy()).round(round_to))}/{ati} ({str(np.array(tn_t).round(round_to))}); "
                f"al {str(np.mean(rl.numpy()).round(round_to))}/{ali} ({str(np.array(tn_l).round(round_to))}); "
            )

            if ma_loss < loss_threshold:
                break

            del tape, ffn, inputs, out, hs, eigs, radius, radius_1


            tf.keras.backend.clear_session()
            tf.keras.backend.clear_session()

        except Exception as e:
            print(e)

    results = {
        'ma_loss': ma_loss,
        'ma_norm': ma_norm,
        'std_ma_norm': std_ma_norm,
        'a_t': np.mean(rt.numpy()),
        'a_l': np.mean(rl.numpy()),
        'loss_t': np.mean(loss_t.numpy()),
        'loss_l': np.mean(loss_l.numpy()),
        'li': li,
        'ni': ni,
        'ali': ali,
        'ati': ati,
        'sti': sti,
        'target_norm': target_norm,
        'tn_t': tn_t,
        'tn_l': tn_l,
        'lsc_comments': comments,
    }
    del rt, rl

    # scales = compare_to_default_scales(width, n_layers, cells)
    # results.update(scales)

    # ffn_weights = equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=cells,
    #                                    path_pretrained=path_pretrained)

    results['final_norm_dec'] = None
    results['final_norms_mean'] = mean_norm
    results['final_norms_std'] = current_std
    results['std_ma_norm'] = std_ma_norm
    results['best_std_ma_norm'] = std_ma_norm

    rec_weights = []
    for ffn in ffns:
        rec_weights.extend(ffn.get_weights())

    allweights = equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=None, path_pretrained=None,
                                      rec_weights=rec_weights)

    del rec_weights, ffns
    if not 'nosgd' in comments:
        del optimizer

    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()

    return allweights, results


def test_1():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='one',
                        type=str, help="String to activate extra behaviors")
    args = parser.parse_args()
    comments = ''
    if 'unbalanced' in args.type:
        comments = 'findLSC_radius_targetnorm:0.5_unbalanced'
    elif 'p5' in args.type:
        comments = 'findLSC_radius_targetnorm:0.5'
    elif 'one' in args.type:
        comments = 'findLSC_radius'

    lruLSC(
        comments=comments, seed=0, stack=4, width=128, classes=2, vocab_size=7,
        maxlen=100, batch_size=32)
    # lruLSC(comments='findLSC_radius_targetnorm:0.5_unbalanced', seed=0, stack=4, width=64, classes=2, vocab_size=7, maxlen=100)
    # lruLSC(comments='findLSC_radius', seed=0, stack=4, width=64, classes=2, vocab_size=7, maxlen=100)
    # lruLSC(comments='test', seed=0, stack=4, width=64, classes=2, vocab_size=7, maxlen=100)
    # equivalence_and_save(width=3, n_layers=2, classes=2, vocab_size=7)
    # save_layer_weights()


def test_2():
    lruLSCffn(stack=2, width=32, classes=2, vocab_size=7, maxlen=8, batch_size=8)


if __name__ == '__main__':
    # test_1()
    test_2()
