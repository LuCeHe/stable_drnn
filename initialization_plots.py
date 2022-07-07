import os, logging
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from GenericTools.stay_organized.mpl_tools import load_plot_settings

mpl = load_plot_settings(mpl=mpl, pd=None)

import matplotlib.pyplot as plt
from tqdm import tqdm
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
    # comments = 'LSC'

    # optimizer properties

    loss_name = 'mse'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'he_normal'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    save_model = False


@ex.capture
@ex.automain
def main(seed, n_neurons, stack, initializer, batch_size, _log):
    exp_dir = os.path.join(*[CDIR, ex.observers[0].basedir])
    images_dir = os.path.join(*[exp_dir, 'images'])

    n_seeds = 10  # 10


    out_dim = 2
    in_dim = 2
    time_steps = 100
    n_rnns = 3
    list_comments = ['LSC', 'dampening:1.', 'randominit', 'lscc']
    # list_comments = ['LSC']

    gs, acts, thrs = [], [], []
    for comments in list_comments:
        gvariances = []
        activities = []
        thresholds = []
        clean_comments = comments.replace('.', 'p').replace(':', '_')
        gv_path = os.path.join(CDIR, 'data', f'gv_{clean_comments}.npy')
        act_path = os.path.join(CDIR, 'data', f'act_{clean_comments}.npy')
        thr_path = os.path.join(CDIR, 'data', f'thr_{clean_comments}.npy')
        print(act_path)
        print(os.path.exists(act_path))
        if not os.path.exists(act_path):
            for seed in tqdm(range(n_seeds)):

                rnns = []
                for i in range(n_rnns):
                    cell = aLSNN(num_neurons=n_neurons, config=comments, name=f'alsnn_{i}', initializer=initializer)
                    rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False, name=f'alsnn_{i}')
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

                activity = trt['alsnn_0']
                threshold = trt['alsnn_0_2']

                var = np.var(dy_dx, axis=(0, 2))
                gvariances.append(var[None])
                activities.append(activity[0][None])
                thresholds.append(threshold[0][None])

            gvariances = np.concatenate(gvariances)
            activities = np.concatenate(activities)
            thresholds = np.concatenate(thresholds)

            for path, numpy in zip([gv_path, act_path, thr_path], [gvariances, activities, thresholds]):
                with open(path, 'wb') as f:
                    np.save(f, numpy)
        else:

            with open(gv_path, 'rb') as f:
                gvariances = np.load(f)
            with open(act_path, 'rb') as f:
                activities = np.load(f)
            with open(thr_path, 'rb') as f:
                thresholds = np.load(f)

        gs.append(gvariances)
        acts.append(activities)
        thrs.append(thresholds)

    colors = [plt.cm.ocean(x) for x in np.linspace(0, 1., n_seeds)]
    colors_thr = [plt.cm.Greens(x) for x in np.linspace(0, 1., 128)]

    Ts = np.arange(1, time_steps + 1)
    fig, axs = plt.subplots(3, len(list_comments), gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))

    for i in range(len(list_comments)):
        axs[1, i].pcolormesh(acts[i][0].T, cmap='Greys')

        for j in range(128):
            axs[2, i].plot(thrs[i][0, ..., j], color=colors_thr[j])

        for var, c in zip(gs[i], colors):
            axs[0, i].plot(Ts, var, color=c)

    for i in range(2):
        for j in range(len(list_comments)):
            axs[i, j].set_xticks([])

    axs[0, 0].set_title('LSC', y=1.2, pad=0)
    axs[0, 1].set_title('naive', y=1.2, pad=0)

    axs[1, 0].set_ylabel('neuron\nactivity')
    axs[2, 0].set_ylabel('neuron\nthreshold')
    axs[0, 0].set_ylabel('gradient\nvariance')
    axs[1, 1].set_xlabel('time')
    axs[0, 1].set_ylim([-1000, 1e10])
    axs[0, 2].set_ylim([0, 10])

    for ax in axs.reshape(-1):
        ax.set_xlim([0, time_steps])

        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    fig.subplots_adjust(top=0.8)
    fig.align_ylabels(axs[:, 0])

    # fig.suptitle(f'Initialization Stats\n{comments}')  # or plt.suptitle('Main title')

    pathplot = os.path.join(CDIR, 'experiments', f'varag'.replace(':', '').replace('.', 'p') + '.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()
