import os, logging
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from GenericTools.stay_organized.mpl_tools import load_plot_settings
from alif_sg.neural_models.custom_lstm import customLSTMcell

mpl = load_plot_settings(mpl=mpl, pd=None)

import matplotlib.pyplot as plt
from tqdm import tqdm
from alif_sg.neural_models.lsnn import aLSNN
from stochastic_spiking.visualization_tools.training_tests import get_test_model

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

seed = 41

# test configuration
stack = 2

# net
net_name = 'maLSNN'
n_neurons = 128
batch_size = 128
loss_name = 'mse'  # categorical_crossentropy categorical_focal_loss contrastive_loss
initializer = 'random_normal'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot random_uniform

n_seeds = 5  # 10
out_dim = 2
in_dim = 2
time_steps = 100
n_rnns = 3
list_comments = ['LSC', 'dampening:1.', 'randominit', 'lscc', 'original', 'LSTM', 'LSTM_LSC']

plot_vargrad = True
plot_binomial = False

gs, acts, thrs = [], [], []

save_folder = os.path.join(CDIR, 'data', initializer)
os.makedirs(save_folder, exist_ok=True)
for comments in list_comments:

    gvariances = []
    activities = []
    thresholds = []
    clean_comments = comments.replace('.', 'p').replace(':', '_')
    gv_path = os.path.join(save_folder, f'gv_{clean_comments}.npy')
    act_path = os.path.join(save_folder, f'act_{clean_comments}.npy')
    thr_path = os.path.join(save_folder, f'thr_{clean_comments}.npy')
    if not os.path.exists(act_path):
        for seed in tqdm(range(n_seeds)):

            rnns = []
            for i in range(n_rnns):
                if not 'LSTM' in comments:
                    cell = aLSNN(num_neurons=n_neurons, config=comments, name=f'alsnn_{i}', initializer=initializer)
                else:
                    cell = customLSTMcell(num_neurons=n_neurons, string_config=comments, name=f'alsnn_{i}')

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


            if not 'LSTM' in comments:
                activity = trt['alsnn_0']
                threshold = trt['alsnn_0_2']
            else:
                print(test_model.output_names)
                activity = trt['alsnn_0']
                threshold = np.zeros_like(trt['alsnn_0'])

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

if plot_vargrad:
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

    for i, c in enumerate(list_comments):
        axs[0, i].set_title(c, y=1.2, pad=0)

    # axs[0, 0].set_title('LSC', y=1.2, pad=0)
    # axs[0, 1].set_title('naive', y=1.2, pad=0)

    axs[1, 0].set_ylabel('neuron\nactivity')
    axs[2, 0].set_ylabel('neuron\nthreshold')
    axs[0, 0].set_ylabel('gradient\nvariance')
    axs[2, -1].set_xlabel('time')
    # axs[0, 1].set_ylim([-1000, 1e10])
    # axs[0, 2].set_ylim([0, 10])

    for ax in axs.reshape(-1):
        ax.set_xlim([0, time_steps])

        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    fig.subplots_adjust(top=0.8)
    fig.align_ylabels(axs[:, 0])

    # fig.suptitle(f'Initialization Stats\n{comments}')  # or plt.suptitle('Main title')

    pathplot = os.path.join(save_folder, 'varag.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()

if plot_binomial:
    from scipy import special as sp
    import numpy as np
    import matplotlib.pyplot as plt

    bound = lambda l, t: sp.binom(t + l + 2, t) / t

    fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace': .25, 'hspace': .1}, figsize=(10, 3))

    T = 10000
    dL = 5

    ts = np.linspace(1, T, 1000)
    y = bound(dL, ts)

    axs[0].plot(ts, y)
    axs[0].set_xlabel(r'$T$')

    T = 5
    dL = 10000

    ls = np.linspace(1, dL, 1000)
    y = bound(ls, T)

    axs[1].plot(ls, y)
    axs[1].set_xlabel(r'$\Delta l$')

    T = 5
    dL = 10000

    ls = np.linspace(1, dL, 1000)
    y = bound(ls, ls / 100)

    axs[2].plot(ls, y)
    axs[2].set_xlabel(r'$\Delta l=100T$')

    axs[0].set_ylabel(r'$\frac{1}{T}\binom{T + \Delta l +2}{T}$')

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    tickers = []
    for i in range(4):
        tickers += np.linspace(10 ** (4 * i + 3), 10 ** (4 * (i + 1) + 3), 50).tolist()
    y_minor = mpl.ticker.FixedLocator(tickers)
    axs[0].yaxis.set_minor_locator(y_minor)

    tickers = []
    for i in range(4):
        tickers += np.linspace(10 ** (4 * i), 10 ** (4 * (i + 1)), 50).tolist()
    y_minor = mpl.ticker.FixedLocator(tickers)
    axs[1].yaxis.set_minor_locator(y_minor)

    #
    # tickers = []
    # for i in range(4):
    #     tickers += np.linspace(10**(44*i+33), 10**(44*(i+1)+33), 50).tolist()
    # y_minor = mpl.ticker.FixedLocator(tickers)
    # axs[2].yaxis.set_minor_locator(y_minor)

    axs[0].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axs[1].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # axs[2].yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    # axs[0].tick_params(axis='y', which='minor')
    pathplot = os.path.join(CDIR, 'experiments', 'subexp.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()
