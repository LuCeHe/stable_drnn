import os, logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment
from alif_sg.neural_models.lsnn import aLSNN
from stochastic_spiking.visualization_tools.training_tests import get_test_model

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
    net_name = 'maLSNN'
    n_neurons = 128
    batch_size = 128

    # LSC dampening:1.
    comments = 'LSC'

    # optimizer properties

    loss_name = 'mse'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    save_model = False


@ex.capture
@ex.automain
def main(comments, seed, n_neurons, stack, initializer, batch_size, _log):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])

    out_dim = 2
    in_dim = 2
    time_steps = 100
    n_rnns = 1
    rnns = []
    for _ in range(n_rnns):
        cell = aLSNN(num_neurons=n_neurons, config=comments, name='alsnn', initializer=initializer)
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False, name='alsnn')
        rnns.append(rnn)

    lin = tf.keras.layers.Input((time_steps, in_dim))
    x = [lin]
    for rnn in rnns:
        x = rnn(x[0])
    readout = tf.keras.layers.Dense(out_dim, name='readout')(x[0])
    model = tf.keras.models.Model(lin, readout)

    tin = tf.convert_to_tensor(np.random.randn(batch_size, time_steps, in_dim))
    tout = tf.convert_to_tensor(np.random.randn(batch_size, time_steps, out_dim))

    # x = tf.constant(3.0)
    with tf.GradientTape() as g:
        g.watch(tin)
        g.watch(tout)
        lout = model(tin)
        loss = tf.keras.losses.MSE(tout, lout)
    dy_dx = g.gradient(loss, tin)

    test_model = get_test_model(model)
    trt = test_model.predict(tin, batch_size=tin.shape[0])
    trt = {name: pred for name, pred in zip(test_model.output_names, trt)}
    print(trt.keys())
    var_voltage = np.var(trt['alsnn_1'], axis=(0, 2))
    activity = trt['alsnn']

    var = np.var(dy_dx, axis=(0, 2))
    print('var g:  ', var[0], var[-1])
    mean = np.mean(dy_dx, axis=(0, 2))
    print('mean g: ', mean[0], mean[-1])
    mean_activity = np.mean(activity, axis=(0, 2))
    print('mean a: ', mean_activity[0], mean_activity[-1])
    # print(mean.shape, var.shape)
    Ts = np.arange(1, len(mean) + 1)
    fig, axs = plt.subplots(2, 2, gridspec_kw={'wspace': .3, 'hspace': .3}, figsize=(10, 3))
    axs[0, 0].plot(Ts, mean)
    axs[0, 0].set_ylabel('gradient\nmean')
    axs[0, 1].plot(Ts, var)
    axs[0, 1].set_ylabel('gradient\nvariance')

    print(activity.shape, var_voltage.shape)
    axs[1, 0].pcolormesh(activity[0].T)
    axs[1, 0].set_ylabel('neuron\n activity')
    axs[1, 1].plot(Ts, var_voltage)
    axs[1, 1].set_ylabel('voltage\nvariance')
    axs[1, 1].set_xlabel('time')


    # for ax in axs:
    #     ax.set_yscale('log')
    #     # ax.set_xscale('log')

    # axs[0, 1].set_yscale('log')
    # axs[1, 1].set_yscale('log')
    fig.subplots_adjust(top=0.8)

    fig.suptitle(f'Initialization Stats\n{comments}')  # or plt.suptitle('Main title')

    pathplot = os.path.join(CDIR, 'experiments', f'varag_{comments}'.replace(':', '').replace('.', 'p') + '.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()
