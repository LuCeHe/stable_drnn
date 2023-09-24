import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from alif_sg.neural_models.recLSC import get_norms
from pyaromatics.keras_tools.esoteric_layers.linear_recurrent_unit import LinearRecurrentUnitCell
from pyaromatics.stay_organized.utils import str2val


def lruLSC(comments='findLSC_radius', lr=1e-3):
    target_norm = str2val(comments, 'targetnorm', float, default=1)
    weight_decay = 1e-4
    rec = 2
    batch_shape = 3
    time_steps = 1
    n_layers = 2
    rand = lambda shape=(rec,), type='r': tf.cast(tf.random.normal(shape), tf.complex64 if type == 'c' else tf.float32)


    adabelief = tfa.optimizers.AdaBelief(lr=lr, weight_decay=weight_decay)
    optimizer = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)


    cells = [LinearRecurrentUnitCell(num_neurons=rec) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

    inputs_t1 = tf.Variable(rand((batch_shape, time_steps, rec)))
    # inputs_t2 = tf.Variable(rand((batch_shape, time_steps, rec)))
    init_states_l1_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]
    init_states_l2_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs_t1)
        # tape.watch(inputs_t2)
        tape.watch(init_states_l1_t1)
        tape.watch(init_states_l2_t1)

        # pick 2 rnns randomly
        # np.random.shuffle(rnns)
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
    print('a_t')
    a_t, loss_t, _ = get_norms(
        tape=tape, lower_states=[states_l2_conc], upper_states=[states_l3_conc], comments=comments,
        target_norm=target_norm
    )
    print(a_t)

    # M_l jacobian of states_l3_conc w.r.t. states_l1_conc
    print('a_l')
    a_l, loss_l, _ = get_norms(
        tape=tape, lower_states=[states_l1_conc], upper_states=[states_l3_conc], comments=comments,
        target_norm=target_norm
    )
    print(a_l)
    mean_loss = loss_t + loss_l
    print(mean_loss)

    trainable_weights = rnn_l2.trainable_weights
    if not 'nosgd' in comments:
        grads = tape.gradient(mean_loss, trainable_weights)
        optimizer.apply_gradients(zip(grads, trainable_weights))
        del grads


if __name__ == '__main__':
    lruLSC()
