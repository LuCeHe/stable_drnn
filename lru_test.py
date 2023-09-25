import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tqdm import tqdm

from alif_sg.neural_models.recLSC import get_norms
from pyaromatics.keras_tools.esoteric_layers.linear_recurrent_unit import LinearRecurrentUnitCell, ResLRUCell
from pyaromatics.stay_organized.utils import str2val


def lruLSC(comments='findLSC_radius', lr=1e-3):
    comments += 'wmultiplier_nosgd'
    target_norm = str2val(comments, 'targetnorm', float, default=0.5)

    rec = 32
    batch_shape = 8
    time_steps = 1
    n_layers = 4
    ts = 100
    tc = n_layers * 2
    round_to = 5
    rand = lambda shape=(rec,): tf.random.normal(shape)

    # cells = [LinearRecurrentUnitCell(num_neurons=rec) for _ in range(n_layers)]
    cells = [ResLRUCell(num_neurons=rec) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

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

        inputs_t1 = tf.Variable(rand((batch_shape, time_steps, rec)))
        init_states_l1_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]
        init_states_l2_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]

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
                    # print(wname)

                    multiplier = 1

                    depth_radius = False
                    if ('C_re' in wname or 'B_re' in wname or 'B_im' in wname or 'C_im' in wname) and pair == 1:
                        depth_radius = True

                    rec_radius = False
                    if 'lambda_nu' in wname:
                        rec_radius = True

                    if depth_radius:
                        local_norm = a_l
                        multiplier = target_norm / local_norm

                    elif rec_radius:
                        local_norm = a_t
                        multiplier = target_norm / local_norm

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

    lsc_results = {
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

    return weights, lsc_results


def test_equivalence():
    rec = 2
    batch_shape = 3
    time_steps = 1
    n_layers = 6
    ts = 2
    tc = n_layers * 2
    round_to = 5
    rand = lambda shape=(rec,): tf.random.normal(shape)

    cells = [ResLRUCell(num_neurons=rec) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(rec,)),
        *rnns,
    ])

    pass


if __name__ == '__main__':
    # lruLSC()
    test_equivalence()
