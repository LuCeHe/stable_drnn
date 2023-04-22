import tensorflow as tf


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
    import numpy as np
    from scipy import special as sp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    L = 10
    batch_size = 6
    dl = L
    T = 100
    tp = np.arange(T)
    dt = T - tp
    n = L + dt - 1
    k = dl

    target_norm = .2
    target_norms = [.5, 1.]

    fig, axs = plt.subplots(1, len(target_norms), figsize=(4, 3), gridspec_kw=dict(wspace=.5, hspace=.3))

    for ax, target_norm in zip(axs, target_norms):
        input_layer = tf.keras.layers.Input((None, 1))
        x = input_layer
        for _ in range(L):
            cell = PascalRNN(num_neurons=1, target_norm=target_norm)
            x = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False)(x)

        it = 2 * tf.random.normal((batch_size, T, 1))
        with tf.GradientTape() as g:
            g.watch(it)
            model = tf.keras.models.Model(input_layer, outputs=x)
            output = model(it)

        dy_dx = g.gradient(output, it)

        # plot different samples of the output with different reds
        oranges = plt.get_cmap('Oranges')
        blues = plt.get_cmap('Blues')
        for i in range(batch_size):
            co = oranges(.3 + i / (batch_size - 1) * .3)
            cb = blues(.4 + i / (batch_size - 1) * .4)

            ax.plot(output[i, :, 0], label='y = PascalRNN(x)', color=cb, linewidth=1)
            ax.plot(dy_dx[i, :, 0], label='dy/dx', color=co, linewidth=2)

        if target_norm == 1.:
            bin = sp.binom(n, k)
            ax.plot(bin + 2e12, label='$\\binom{\,L + \\Delta t - 1}{\,\\Delta l}$', color='g', linewidth=2)

        ax.set_title(f'$\\rho_p = {target_norm}$', fontsize=14, y=1.2)
        ax.set_xlabel('t')
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=c, lw=4, label=n)
        for c, n in [
            ['g', '$\\binom{\,L + \\Delta t - 1}{\,\\Delta l}$'],
            [oranges(.6), 'dy/dx'],
            [blues(.6), 'y = PascalRNN$_{\\rho_p}$(x)'],
        ]
    ]
    plt.legend(ncols=3, handles=legend_elements, loc='lower center', bbox_to_anchor=(-0.3, -.4))

    plot_filename = f'../experiments/pascal.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()
