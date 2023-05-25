import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from pyaromatics.stay_organized.utils import setReproducible
from pyaromatics.keras_tools.esoteric_losses.loss_redirection import get_loss
from keras_tools.esoteric_initializers.out_initializer import OutInitializer
# from alif_sg.generate_data.task_redirection import Task, language_tasks
# from alif_sg.neural_models.custom_lstm import customLSTMcell
# from sg_design_lif.neural_models.full_model import build_model
from stochastic_spiking.generate_data.task_redirection import language_tasks

# mpl = load_plot_settings(mpl=mpl, pd=None)

from stochastic_spiking.visualization_tools.training_tests import get_test_model

from scipy import special as sp
from scipy.optimize import curve_fit

bound = lambda l, t: sp.binom(t + l + 2, t) / t
bound = lambda l, t: sp.binom(t + l, t)

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
    plot_binomial = True
    plot_activity = False
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

    if plot_vargrad or plot_activity:
        gs, acts, thrs, cvolts = get_data(n_neurons, time_steps, list_comments, save_folder, n_seeds, initializer,
                                          stack)

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

        fontsize = 18
        linewidth = 2
        # fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace': .5, 'hspace': .1}, figsize=(6, 3))
        fig, axs = plt.subplot_mosaic([['i)', 'ii)', 'iii)']], layout='constrained', figsize=(6, 3))

        for label, ax in axs.items():
            ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize)
            ax.set_yscale('log')

            if label == 'i)':
                T, dL = 10000, 5

                ts = np.linspace(1, T, 1000)
                y = bound(dL, ts)

                ax.plot(ts, y, color='#018E32', linewidth=linewidth)
                ax.set_xlabel(r'$T$', fontsize=fontsize)
                ax.set_yticks([1e8, 1e16])

                # axs[0].set_ylabel(r'$\frac{1}{T}\binom{T + \Delta l +2}{T}$', fontsize=fontsize + 6)
                ax.set_ylabel('Number of descent paths\nin rectangular grid', fontsize=fontsize * 1.1, labelpad=20)

            elif label == 'ii)':

                T, dL = 5, 100

                ls = np.linspace(1, dL, 1000)
                y = bound(ls, T)

                ax.plot(ls, y, color='#018E32', linewidth=linewidth)
                ax.set_xlabel(r'$\Delta l$', fontsize=fontsize)
                ax.set_yticks([1e4, 1e7])

            elif label == 'iii)':

                T = 10000
                ts = np.linspace(1, T, 1000)
                y = bound(ts / 100, ts)

                ax.plot(ts, y, color='#018E32', linewidth=linewidth)
                ax.set_xlabel(r'$100\Delta l=T$', fontsize=fontsize)
                ax.set_yticks([1e110, 1e220])

            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

            xlabels = [f'{int(x / 1000)}K' if x > 1000 else int(x) for x in ax.get_xticks()]
            ax.set_xticklabels(xlabels)
            ax.minorticks_off()
            ax.tick_params(axis='both', which='major', labelsize=fontsize * .9)
            ax.yaxis.tick_right()

        # axs[0].tick_params(axis='y', which='minor')
        pathplot = os.path.join(CDIR, 'experiments', 'subexp.pdf')
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


def get_norm(td, norm_pow, numpy=False):
    if norm_pow == -1:
        norms = tf.reduce_max(tf.reduce_sum(tf.abs(td), axis=2), axis=-1)

    elif norm_pow == 1:
        norms = tf.reduce_max(tf.reduce_sum(tf.abs(td), axis=1), axis=-1)

    elif norm_pow == 2:
        norms = tf.linalg.svd(td, compute_uv=False)[..., 0]

    elif norm_pow == 0:
        norms = tf.abs(tf.linalg.eigvals(td)[..., 0])

    else:
        raise ValueError('norm_pow must be -1, 1 or 2')

    if numpy:
        return norms.numpy()
    return norms


def plot_matrix_norms():
    batch_size = 1
    n = 1000
    ns = np.linspace(10, 1000, 20)
    norm_pows = [1, 2, -1, 0]
    data = {k: [] for k in norm_pows}

    for n in tqdm(ns):
        n = int(n)
        # td = tf.random.normal((batch_size, n, n))
        td = tf.random.uniform((batch_size, n, n), minval=-np.sqrt(3), maxval=np.sqrt(3))
        # norms = get_norm(td, 2)
        # print(n, norms[0])
        for norm_pow in norm_pows:
            norms = get_norm(td, norm_pow)
            # den = np.sqrt(n) if norm_pow == 2 else n
            data[norm_pow].append(norms[0])
            den = 1
            # print(norm_pow, norms / den)

    fig, axs = plt.subplots(1, 1, gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))

    for norm_pow in norm_pows:
        axs.plot(ns, data[norm_pow], label=f'norm_pow={norm_pow}')

    axs.legend()
    plt.show()
    # td = tf.random.normal((batch_size, n, n))
    # print(tf.reduce_mean(td), tf.math.reduce_variance(td))


class GetNorm(tf.keras.layers.Layer):
    def __init__(self, init_tensor, norm_pow, target_norm=1, **kwargs):
        super().__init__(**kwargs)
        self.init_tensor = init_tensor
        self.norm_pow = norm_pow
        self.target_norm = target_norm

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w', shape=self.init_tensor.shape,
            initializer=OutInitializer(init_tensor=self.init_tensor),
            trainable=True
        )

        self.built = True

    def call(self, inputs, **kwargs):
        norms = get_norm(self.w, self.norm_pow)
        loss = tf.reduce_mean(tf.square(norms - self.target_norm))
        self.add_loss(loss)
        self.add_metric(loss, name='cost', aggregation='mean')

        return inputs


def plot_matrix_norms_2():
    n = 1000

    # # td = tf.random.normal((n, n))
    td = tf.random.uniform((n, n), minval=-np.sqrt(3), maxval=np.sqrt(3))

    layer = GetNorm(td, norm_pow=2, target_norm=1)
    layer.build((2, 2))
    ins = tf.keras.layers.Input(shape=(2, 2))
    outs = layer(ins)
    model = tf.keras.Model(inputs=ins, outputs=outs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss=None, optimizer=optimizer)

    fake_data = tf.random.normal((2, 2))
    history = model.fit(fake_data, fake_data, epochs=200, verbose=1)

    fig, axs = plt.subplots(1, 2, gridspec_kw={'wspace': .3, 'hspace': .1}, figsize=(10, 5))
    w = layer.w.numpy().flatten()
    axs[0].hist(w, 50, density=True, facecolor='g', alpha=0.75)
    axs[1].plot(history.history['loss'])
    plt.show()


if __name__ == '__main__':
    initialization_tests()
