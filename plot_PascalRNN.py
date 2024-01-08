import tensorflow as tf

from drnn_stability.neural_models.recLSC import load_LSC_model
from lif_stability.config.config import default_config
from lif_stability.neural_models.full_model import build_model


class PascalRNN(tf.keras.layers.Layer):

    def __init__(self, num_neurons=None, target_norm=1., **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_neurons=num_neurons, target_norm=target_norm)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

    def call(self, inputs, states, **kwargs):
        output = self.target_norm * inputs + self.target_norm * states[0]
        new_state = (output,)
        return output, new_state


if __name__ == '__main__':
    import os
    import numpy as np
    from scipy import special as sp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    CDIR = os.path.dirname(__file__)
    EXPSDIR = os.path.abspath(os.path.join(CDIR, '..', 'experiments'))


    def build_pascal_model(L, target_norm=1.):
        input_layer = tf.keras.layers.Input((None, 1))
        x = input_layer
        for _ in range(L):
            cell = PascalRNN(num_neurons=1, target_norm=target_norm)
            x = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False)(x)

        model = tf.keras.models.Model(input_layer, outputs=x)
        return model


    target_norm = .2
    target_norms = [.5, 1.]

    # nets = ['pascal', 'GRU', ]
    nets = ['pascal', ]
    fig, axs = plt.subplots(len(nets), len(target_norms), figsize=(4, 3), gridspec_kw=dict(wspace=.4, hspace=.5))

    net_name = 'GRU'
    for i, net_name in enumerate(nets):
        if net_name == 'pascal':
            build_model_ = build_pascal_model
            L = 10
            batch_size = 6
            T = 100

            it = 2 * tf.random.normal((batch_size, T, 1))

        else:
            from pyaromatics.keras_tools.esoteric_tasks.time_task_redirection import Task

            comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2'
            stack, batch_size, embedding, n_neurons, lr = default_config(
                None, None, None, None, None, 'heidelberg', net_name, setting='LSC'
            )
            path_1 = rf'D:\work\drnn_stability\good_experiments\pmodels\pretrained_s0_{net_name}_radius_heidelberg_stack7.h5'
            path_05 = rf'D:\work\drnn_stability\good_experiments\pmodels\pretrained_s0_{net_name}_radius_heidelberg_stack7_tn0p5.h5'
            # model = load_LSC_model(path_1)

            train_task_args = dict(timerepeat=2, epochs=1, batch_size=batch_size, steps_per_epoch=2,
                                   name='heidelberg', train_val_test='train', maxlen=100, comments='')
            gen_train = Task(**train_task_args)
            it = gen_train.__getitem__()[0]
            L = 7
            T = it[0].shape[1]


            def build_model_(L, target_norm=1):
                if target_norm == 1:
                    states_model = load_LSC_model(path_1)
                elif target_norm == .5:
                    states_model = load_LSC_model(path_05)
                else:
                    raise ValueError
                weights = states_model.get_weights()

                model_args = dict(
                    task_name='heidelberg', net_name=net_name, n_neurons=n_neurons,
                    lr=1., stack=7, loss_name='sparse_categorical_crossentropy',
                    embedding=None, optimizer_name='AdamW', lr_schedule='',
                    weight_decay=0, clipnorm=None, initializer='ones', comments=comments,
                    in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
                    n_out=gen_train.out_dim, final_epochs=gen_train.epochs, seed=0,
                )
                model = build_model(**model_args)
                model.set_weights(weights)
                return model

        dl = L
        tp = np.arange(T)
        dt = T - tp
        n = L + dt - 1
        k = dl

        for j, target_norm in enumerate(target_norms):
            print(f'Plotting for target_norm = {target_norm}')
            path_data = os.path.join(EXPSDIR, rf'{net_name}_tn{target_norm}_L{L}.npz')
            if not os.path.exists(path_data):
                with tf.GradientTape() as g:
                    if isinstance(it, list) or isinstance(it, tuple):
                        it = [tf.convert_to_tensor(t, dtype=tf.float32) for t in it]
                    else:
                        it = tf.convert_to_tensor(it, dtype=tf.float32)
                    g.watch(it)
                    model = build_model_(L, target_norm=target_norm)
                    output = model(it)

                dy_dx = g.gradient(output, it) if net_name == 'pascal' else g.gradient(output, it[0])

                # save the output and dy_dx
                np.savez_compressed(path_data, output=output, dy_dx=dy_dx)
            # load the output and dy_dx
            data = np.load(path_data)
            # output, dy_dx, it = data['output'], data['dy_dx'], data['it']
            output, dy_dx = data['output'], data['dy_dx']

            # plot different samples of the output with different reds
            oranges = plt.get_cmap('Oranges')
            blues = plt.get_cmap('Blues')
            greens = plt.get_cmap('Greens')
            g = greens(.6)
            ax = axs[i, j]
            for b in range(batch_size):
                cb = blues(.4 + b / (batch_size - 1) * .4)
                ax.plot(output[b, :, 0], label='y = PascalRNN(x)', color=cb, linewidth=1)

            for b in range(batch_size):
                co = oranges(.3 + b / (batch_size - 1) * .3)
                ax.plot(np.abs(dy_dx[b, :, 0]), label='dy/dx', color=co, linewidth=2)

            if target_norm == 1.:
                bin = sp.binom(n, k)
                # bin = bin / np.amax(bin) * np.amax(np.abs(dy_dx))*1.05/3
                epsilon = 2e12 if L == 10 else 0
                m = bin + 2e12 if net_name == 'pascal' else \
                    bin / np.amax(bin) * np.amax(np.abs(dy_dx)) * 1.05 / 5 + 1.

                ax.plot(m, label='$\\binom{\,L + \\Delta t - 1}{\,\\Delta l}$', color='g', linewidth=2)
            elif target_norm == .5:
                m = np.amax(np.abs(dy_dx)) * 1.05
                m = m if net_name == 'pascal' else m / 2
                ax.plot([0, T], [m, m], color=g, linewidth=2)

            if i == 0:
                ax.set_title(f'$\\rho = {target_norm}$', fontsize=14, y=1.2)
            elif i == len(nets) - 1:
                ax.set_xlabel('t')

            if j == 0:
                ax.set_ylabel('amplitude')


            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

    fig.align_ylabels(axs[:, 0])

    # write the names of the nets on the left side and rotated vertically
    for i, net_name in enumerate(nets):
        ax = axs[i, 0]
        ax.text(-0.65, 0.5, net_name.replace('pascal','PascalRNN'),
                fontsize=14, transform=ax.transAxes, rotation=90, va='center', ha='center')

    line = plt.Line2D([-.05,.9],[.46,.46], transform=fig.transFigure, color="black", linewidth=.5)
    fig.add_artist(line)

    legend_elements = [
        Line2D([0], [0], color=c, lw=4, label=n)
        for c, n in [
            [g, '$c_1$'],
            ['g', '$c_1\\binom{\,L + \\Delta t}{\,\\Delta t}+c_2$'],
            [oranges(.6), '|dy/dx|'],
            [blues(.6), 'y = Net(x)'],
        ]
    ]
    plt.legend(ncols=2, handles=legend_elements, loc='lower center', bbox_to_anchor=(-0.3, -1.2))

    plot_filename = f'../experiments/pascal.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()
