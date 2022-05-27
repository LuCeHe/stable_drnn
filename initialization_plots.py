import os, logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment
from alif_sg.neural_models.lsnn import aLSNN

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-initalif', base_dir=CDIR, seed=11)
logger = logging.getLogger('mylogger')

language_tasks = ['ptb', 'wiki103', 'wmt14', 'time_ae_merge', 'monkey', 'wordptb']


@ex.config
def config():
    # environment properties
    seed = 41

    # test configuration
    stack = 2

    # net
    # mn_aLSNN_2 mn_aLSNN_2_sig LSNN maLSNN spikingPerformer smallGPT2 aLSNN_noIC spikingLSTM
    net_name = 'maLSNN'
    n_neurons = 100
    batch_size = 128
    comments = ''

    # optimizer properties

    loss_name = 'mse'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    save_model = False


@ex.capture
@ex.automain
def main(comments, seed, n_neurons, stack, initializer, batch_size, _log):
    out_dim = 2
    in_dim = 2
    lin = tf.keras.layers.Input((100, in_dim))
    cell = aLSNN(num_neurons=n_neurons)
    alsnn = tf.keras.layers.RNN(cell, return_sequences=True, name='encoder', stateful=False)(lin)[0]

    readout = tf.keras.layers.Dense(out_dim)(alsnn)
    model = tf.keras.models.Model(lin, readout)

    tin = tf.convert_to_tensor(np.random.randn(batch_size, 100, in_dim))
    tout = tf.convert_to_tensor(np.random.randn(batch_size, 100, out_dim))

    # x = tf.constant(3.0)
    with tf.GradientTape() as g:
        g.watch(tin)
        g.watch(tout)
        lout = model(tin)
        loss = tf.keras.losses.MSE(tout, lout)
    dy_dx = g.gradient(loss, tin)

    var = np.var(dy_dx, axis=(0, 2))
    print('var:  ', var[0], var[-1])
    mean = np.mean(dy_dx, axis=(0, 2))
    print('mean: ', mean[0], mean[-1])
    print(mean.shape, var.shape)
    Ts = np.arange(1, len(mean)+ 1)
    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': .1, 'hspace': .05}, figsize=(10, 3))
    axs[0].plot(Ts, mean)
    axs[1].plot(Ts, var)

    for ax in axs:
        ax.set_yscale('log')
        # ax.set_xscale('log')

    plt.show()