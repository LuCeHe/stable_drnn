import os, logging
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from tqdm import tqdm
import matplotlib.pyplot as plt

from GenericTools.keras_tools.esoteric_layers.surrogated_step import ChoosePseudoHeaviside
from GenericTools.stay_organized.mpl_tools import load_plot_settings
from GenericTools.stay_organized.utils import setReproducible, str2val
from GenericTools.keras_tools.esoteric_losses.loss_redirection import get_loss
from alif_sg.generate_data.task_redirection import Task, language_tasks
from alif_sg.neural_models.custom_lstm import customLSTMcell
from sg_design_lif.neural_models.full_model import build_model

mpl = load_plot_settings(mpl=mpl, pd=None)

from alif_sg.neural_models.lsnn import aLSNN
from stochastic_spiking.visualization_tools.training_tests import get_test_model

from scipy import special as sp
from scipy.optimize import curve_fit

bound = lambda l, t: sp.binom(t + l + 2, t) / t

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')


def get_data(n_neurons, time_steps, list_comments, save_folder, n_seeds, initializer, stack):
    seed = 41

    # net
    net_name = 'aLSNN'
    batch_size = 128
    task_name = 'heidelberg'  # random wordptb heidelberg
    embedding = 'learned:None:None:{}'.format(n_neurons) if task_name in language_tasks else False

    if task_name == 'random':
        out_dim = 4
        in_dim = 2
        tin = np.random.randn(batch_size, time_steps, in_dim)
        # tout = np.random.randn(batch_size, time_steps, out_dim)
        tout = tf.random.uniform(shape=(batch_size, time_steps), maxval=out_dim, dtype=tf.int32, seed=10)

        loss_name = 'sparse_categorical_crossentropy'  # mse
    else:
        gen_train = Task(timerepeat=2, epochs=0, batch_size=batch_size, steps_per_epoch=0,
                         name=task_name, train_val_test='train', maxlen=time_steps, comments='', lr=0)
        (tin, tout), = gen_train.__getitem__()

        out_dim = gen_train.out_dim
        in_dim = gen_train.in_dim
        time_steps = gen_train.out_len
        loss_name = 'sparse_categorical_crossentropy'

    gs, acts, thrs, cvolts = [], [], [], []

    tin = tf.cast(tf.convert_to_tensor(tin, dtype=tf.float32), tf.float32)
    tout = tf.convert_to_tensor(tf.cast(tout, tf.float32), dtype=tf.float32)

    os.makedirs(save_folder, exist_ok=True)
    for comments in list_comments:

        gvariances = []
        activities = []
        thresholds = []
        centered_voltages = []
        clean_comments = comments.replace('.', 'p').replace(':', '_')
        gv_path = os.path.join(save_folder, f'gv_{clean_comments}_{task_name}.npy')
        act_path = os.path.join(save_folder, f'act_{clean_comments}_{task_name}.npy')
        thr_path = os.path.join(save_folder, f'thr_{clean_comments}_{task_name}.npy')
        cv_path = os.path.join(save_folder, f'cv_{clean_comments}_{task_name}.npy')
        if not os.path.exists(act_path):
            for seed in tqdm(range(n_seeds)):
                setReproducible(seed)
                netn = 'cLSTM' if 'cLSTM' in comments else net_name
                model = build_model(
                    task_name=task_name, net_name=netn, n_neurons=n_neurons,
                    lr=0, stack=stack, loss_name=loss_name,
                    embedding=embedding, optimizer_name='SWAAdaBelief', lr_schedule='',
                    weight_decay=.1, clipnorm=None, initializer=initializer, comments=comments,
                    in_len=time_steps, n_in=in_dim, out_len=time_steps,
                    n_out=out_dim, final_epochs=0,
                )
                # model.summary()
                loss_fn = get_loss(loss_name)

                # x = tf.constant(3.0)
                with tf.GradientTape() as g:
                    g.watch(tin)
                    g.watch(tout)
                    inlayer = tin
                    # if task_name in language_tasks:
                    #     # lns = [layer.name for layer in model.layers]
                    #
                    #     ln = [layer.name for layer in model.layers if 'encoder_0_0' in layer.name][0]
                    #     # inlayer = model.get_layer(ln).output
                    #     inlayer = model.get_layer(ln).input
                    #     # inlayer = model.get_layer(ln)
                    #     raise ValueError('This code has to be finished')

                    lout = model([tin, tout])
                    loss = loss_fn(tout, lout)
                dy_dx = g.gradient(loss, inlayer)
                # print('hey!')
                # print(dy_dx)
                # print(loss)

                test_model = get_test_model(model)
                trt = test_model.predict([tin, tout], batch_size=tin.shape[0])
                trt = {name: pred for name, pred in zip(test_model.output_names, trt)}

                if not 'LSTM' in comments:
                    activity = trt['encoder_0_0']
                    threshold = trt['encoder_0_0_2']
                    cv = trt['encoder_0_0_3']
                else:
                    activity = trt['encoder_0_0']
                    threshold = np.zeros((1,))
                    cv = np.zeros((1,))

                if not task_name in language_tasks:
                    var = np.var(dy_dx, axis=(0, 2))
                else:
                    var = np.zeros((1,))
                gvariances.append(var[None])
                activities.append(activity[0][None])
                thresholds.append(threshold[0][None])
                centered_voltages.append(cv[None])

            gvariances = np.concatenate(gvariances)
            activities = np.concatenate(activities)
            thresholds = np.concatenate(thresholds)
            centered_voltages = np.concatenate(centered_voltages)

            for path, numpy in zip(
                    [gv_path, act_path, thr_path, cv_path],
                    [gvariances, activities, thresholds, centered_voltages]
            ):
                with open(path, 'wb') as f:
                    np.save(f, numpy)
        else:

            with open(gv_path, 'rb') as f:
                gvariances = np.load(f)
            with open(act_path, 'rb') as f:
                activities = np.load(f)
            with open(thr_path, 'rb') as f:
                thresholds = np.load(f)
            with open(cv_path, 'rb') as f:
                centered_voltages = np.load(f)

        gs.append(gvariances)
        acts.append(activities)
        thrs.append(thresholds)
        cvolts.append(centered_voltages)

    return gs, acts, thrs, cvolts


def initialization_tests():
    plot_vargrad = False
    plot_binomial = False
    plot_activity = True
    test_adaptive_pseudod = False

    # test configuration
    stack = 2
    n_neurons = 128
    n_seeds = 10  # 10
    time_steps = 100
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot random_uniform
    save_folder = os.path.join(CDIR, 'data', initializer)

    # list_comments = ['LSC1', 'lsc1', 'LSC2', '', 'randominit', 'cLSTM', 'veryrandom']
    list_comments = ['LSC1', 'lsc1', 'LSC2', '', 'randominit', 'cLSTM']

    gs, acts, thrs, cvolts = get_data(n_neurons, time_steps, list_comments, save_folder, n_seeds, initializer, stack)

    if plot_vargrad:
        colors = [plt.cm.ocean(x) for x in np.linspace(.2, .8, n_seeds)]
        colors_thr = [plt.cm.Greens(x) for x in np.linspace(0, 1., 128)]

        Ts = np.arange(1, time_steps + 1)
        fig, axs = plt.subplots(3, len(list_comments), gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))

        for i in range(len(list_comments)):
            axs[1, i].pcolormesh(acts[i][0].T, cmap='Greys')

            if not 'LSTM' in list_comments[i]:
                for j in range(min(n_neurons, 128)):
                    axs[2, i].plot(thrs[i][0, ..., j], color=colors_thr[j])

            for var, c in zip(gs[i], colors):
                # print(Ts.shape, var.shape)
                axs[0, i].plot(Ts, var, color=c)

            y_bound = bound(stack, Ts) / n_neurons
            axs[0, i].plot(Ts, list(reversed(y_bound)), '--', color='b')

        for i in range(2):
            for j in range(len(list_comments)):
                axs[i, j].set_xticks([])

        for i, c in enumerate(list_comments):
            axs[0, i].set_title(c, y=1.2, pad=0)
            axs[0, i].set_yscale('log')

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

        fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace': .25, 'hspace': .1}, figsize=(10, 3))

        T = 10000
        dL = 5

        ts = np.linspace(1, T, 1000)
        y = bound(dL, ts)

        axs[0].plot(ts, y)
        axs[0].set_xlabel(r'$T$')

        T = 5
        dL = 1000

        ls = np.linspace(1, dL, 1000)
        y = bound(ls, T)

        axs[1].plot(ls, y)
        axs[1].set_xlabel(r'$\Delta l$')

        T = 10000
        ts = np.linspace(1, T, 1000)
        y = bound(ts / 100, ts)

        axs[2].plot(ts, y)
        axs[2].set_xlabel(r'$100\Delta l=T$')

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
        for i in range(3):
            tickers += np.linspace(10 ** (3 * i + 3), 10 ** (3 * (i + 1) + 3), 50).tolist()
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

    if plot_activity:
        plot_act(cvolts, list_comments)

    if test_adaptive_pseudod:
        test_adaptsg()


def plot_act(cvolts, list_comments):
    # colors = [plt.cm.ocean(x) for x in np.linspace(0, 1., n_seeds)]
    # colors_thr = [plt.cm.Greens(x) for x in np.linspace(0, 1., 128)]

    fig, axs = plt.subplots(1, len(list_comments), gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))

    # Here you give the initial parameters for a,b,c which Python then iterates over
    # to find the best fit
    # popt, pcov = curve_fit(double_exp, x, y, p0=(1.0, 1.0, 1.0, 1.0))
    for i, c in enumerate(list_comments):
        cv = cvolts[i].flatten()
        # the histogram of the data
        axs[i].set_title(c)
        n, bins, patches = axs[i].hist(cv, 1000, density=True, facecolor='g', alpha=0.5)
        popt, pcov = curve_fit(double_exp, bins[:-1], n, p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .5, .5, .5, .5, .5))
        popt = [-4.70412568e-04, 2.72106219e+00, - 6.93245581e-08, 8.43832680e+00,
                5.14831068e-02, - 1.24047899e-01, 9.01356952e-02, - 1.05512663e-01,
                - 2.60721870e-01, - 9.96660911e-01, 1.44756129e+00]
        popt = [abs(p) for p in popt]
        # axs[i].plot(bins[:-1], double_exp(bins[:-1], *popt), '--', color='r')
        maxval = max(popt[0] + popt[4], popt[2] + popt[7])
        axs[i].plot(bins[:-1], double_exp(bins[:-1], *popt) / maxval, '--', color='r')
        print(c, *popt)

    pathplot = os.path.join(EXPERIMENTS, 'vdist.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()


def test_adaptsg():
    params = [-0.20201285957912918, 2.5118566166639584, -0.6673356614771232, 1.2510880363762846,
              -0.049519468893990996,
              0.004120122160092256, 2.0072643043174256, 0.8743848390304889, 1.5061652880205298, 1.1392061683504318,
              0.30571907616976585]

    config = 'eppseudod'
    for l, v in zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm'], params):
        config += f'_{l}:{v}'
    print(config)
    v_sc = tf.random.uniform((2, 3))
    ChoosePseudoHeaviside(v_sc, config=config, sharpness=1, dampening=1)


def double_exp(x, a, b, c, d, e, f, g, h, i, l, m):
    a, b, c, d, e, f, g, h, i, l, m = abs(a), abs(b), abs(c), abs(d), abs(e), abs(f), abs(g), abs(h), abs(i), abs(l), m
    x = x + m
    return \
        (a * np.exp(b * x) + e * 1 / (1 + f * np.abs(x) ** (1 + i))) * np.heaviside(-x, .5) \
        + (c * np.exp(-d * x) + h * 1 / (1 + g * np.abs(x) ** (1 + l))) * np.heaviside(x, .5)  # one of best so far


def adapt_sg_shape(data_generator, model, comments):

    (tin, tout), = data_generator.__getitem__()

    test_model = get_test_model(model)
    trt = test_model.predict([tin, tout], batch_size=tin.shape[0])
    trt = {name: pred for name, pred in zip(test_model.output_names, trt)}

    activity_names = [k for k in trt.keys() if k.endswith('_3') and k.startswith('encoder')]
    for i, k in enumerate(activity_names):
        cv = trt[k].flatten()

        n, bins, patches = plt.hist(cv, 1000, density=True, facecolor='g', alpha=0.5)
        try:
            popt, _ = curve_fit(double_exp, bins[:-1], n, p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, .5, .5, .5, .5, .5))
            print('\n\n            Adapted SG!\n\n')
        except Exception as e:
            print('\n\n            Non Adapted SG error!\n\n')
            print(e)
            popt = [
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 0
            ]

        comments += '_eppseudod'
        for l, v in zip(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm'], popt):
            comments += f'_{l}{i}:{v}'
    return comments


if __name__ == '__main__':
    initialization_tests()
