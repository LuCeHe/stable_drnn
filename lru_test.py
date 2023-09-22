import tensorflow as tf
import numpy as np

from alif_sg.neural_models.recLSC import get_norms
from pyaromatics.keras_tools.esoteric_layers.linear_recurrent_unit import LinearRecurrentUnitCell
from pyaromatics.stay_organized.utils import str2val


def test_4(comments='findLSC_radius'):
    target_norm = str2val(comments, 'targetnorm', float, default=1)

    rec = 2
    batch_shape = 3
    time_steps = 1
    n_layers = 2
    rand = lambda shape=(rec,), type='r': tf.cast(tf.random.normal(shape), tf.complex64 if type == 'c' else tf.float32)

    cells = [LinearRecurrentUnitCell(num_neurons=rec) for _ in range(n_layers)]
    rnns = [tf.keras.layers.RNN(cell, return_sequences=True, return_state=True) for cell in cells]

    inputs_t1 = tf.Variable(rand((batch_shape, time_steps, rec)))
    # inputs_t2 = tf.Variable(rand((batch_shape, time_steps, rec)))
    init_states_l1_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]
    init_states_l2_t1 = [tf.Variable(rand((batch_shape, 2 * rec))), tf.Variable(rand((batch_shape, 2 * rec)))]

    with tf.GradientTape(persistent=True) as tape:
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
    a_t = get_norms(
        tape=tape, lower_states=[states_l1_conc], upper_states=[states_l2_conc], comments=comments,
        target_norm=target_norm
    )
    print(a_t)

    # M_l jacobian of states_l3_conc w.r.t. states_l1_conc
    print('a_l')
    a_l = get_norms(
        tape=tape, lower_states=[states_l1_conc], upper_states=[states_l3_conc], comments=comments,
        target_norm=target_norm
    )
    print(a_l)


if __name__ == '__main__':
    test_4()
