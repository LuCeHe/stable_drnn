import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm

from alif_sg.neural_models.recLSC import get_norms
from alif_sg.tools.admin_model_removal import get_pretrained_file
from pyaromatics.keras_tools.esoteric_layers.linear_recurrent_unit import LinearRecurrentUnitCell, ResLRUCell, ResLRUFFN
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


def lruLSC(comments='findLSC_radius', seed=0, stack=4, width=32, classes=2, vocab_size=7, maxlen=1):
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

    target_norm = str2val(comments, 'targetnorm', float, default=1)

    batch_shape = 8
    time_steps = 1

    n_layers = int(stack)
    ts = 50 if 'test' in comments else 500  # number of pretraining steps
    tc = n_layers * 2  # time constant for the moving averages
    round_to = 5
    rand = lambda shape=(width,): tf.random.normal(shape)

    if 'onlyloadpretrained' in comments:
        ts = 10
    else:
        comments += 'wmultiplier'

    cells = [ResLRUCell(num_neurons=width) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

    if not ffn_stem is None:
        rnns_nrs = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in cells]
        rnn_stem = tf.keras.models.Sequential(rnns_nrs)
        rnn_stem.build((None, None, width))
        rnn_stem.set_weights(ffn_stem.get_weights())

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

        inputs_t1 = tf.Variable(rand((batch_shape, time_steps, width)))
        init_states_l1_t1 = [
            tf.Variable(rand((batch_shape, 2 * width))),
            tf.Variable(rand((batch_shape, 2 * width)))
        ]
        init_states_l2_t1 = [
            tf.Variable(rand((batch_shape, 2 * width))),
            tf.Variable(rand((batch_shape, 2 * width)))
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
            target_norm=target_norm
        )

        # M_l jacobian of states_l3_conc w.r.t. states_l1_conc
        a_l, loss_l, _ = get_norms(
            tape=tape, lower_states=[states_l1_conc], upper_states=[states_l3_conc], comments=comments,
            target_norm=target_norm
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

        wnames_2 = [weight.name for weight in rnn_l2.weights]
        wnames_1 = [weight.name for weight in rnn_l1.weights]

        weights_2 = rnn_l2.trainable_weights
        weights_1 = rnn_l1.trainable_weights

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
                        tn = target_norm
                        if target_norm == 0.5 and 'unbalanced' in comments:
                            tn = n_layers / (n_layers + maxlen)

                        multiplier = tn / local_norm

                    elif rec_radius:
                        local_norm = a_t
                        tn = target_norm
                        if target_norm == 0.5 and 'unbalanced' in comments:
                            tn = maxlen / (n_layers + maxlen)

                        multiplier = tn / local_norm

                    m = tf.reduce_mean(multiplier).numpy()
                    m = np.clip(m, 0.95, 1.05)

                    w = m * w
                    new_weights.append(w)

                rnn.set_weights(new_weights)

        pbar.set_description(
            f"Step {t}; "
            f"loss {str(np.array(ma_loss).round(round_to))}/{li}; "
            f"mean norms {np.array(ma_norm).round(round_to)}/{ni}; "
            f"ma std norms {str(np.array(std_ma_norm).round(round_to))}/{sti}; "
            f"at {str(np.mean(a_t.numpy()).round(round_to))}/{ati}; "
            f"al {str(np.mean(a_l.numpy()).round(round_to))}/{ali}; "
        )

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
    }

    ffn_weights = equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=cells,
                                       path_pretrained=path_pretrained)

    results['final_norm_dec'] = None
    results['final_norms_mean'] = mean_norm
    results['final_norms_std'] = current_std
    results['std_ma_norm'] = std_ma_norm
    results['best_std_ma_norm'] = std_ma_norm
    return ffn_weights, results


def equivalence_and_save(comments, width, n_layers, classes, vocab_size, cells=None, path_pretrained=None):
    if cells is None:
        cells = [ResLRUCell(num_neurons=width) for _ in range(n_layers)]

    rnns = [tf.keras.layers.RNN(cell, return_sequences=True) for cell in cells]

    rnn_stem = tf.keras.models.Sequential(rnns)

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


if __name__ == '__main__':
    lruLSC(comments='findLSC_radius_targetnorm:0.5_unbalanced', seed=0, stack=4, width=32, classes=2, vocab_size=7, maxlen=100)
    # equivalence_and_save(width=3, n_layers=2, classes=2, vocab_size=7)
    # save_layer_weights()