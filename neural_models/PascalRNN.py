import tensorflow as tf


class PascalRNN(tf.keras.layers.Layer):

    def __init__(self, num_neurons=None, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_neurons=num_neurons)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

    def call(self, inputs, states, **kwargs):
        output = inputs + states[0]
        new_state = (output,)
        return output, new_state


if __name__ == '__main__':
    import numpy as np
    from scipy import special as sp
    import matplotlib.pyplot as plt

    L = 10
    dl = L
    T = 100
    tp = np.arange(T)
    dt = T - tp
    n = L + dt - 1
    k = dl

    input_layer = tf.keras.layers.Input((None, 1))
    x = input_layer
    for _ in range(L):
        cell = PascalRNN(num_neurons=1)
        x = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False)(x)

    it = tf.ones((1, T, 1))
    with tf.GradientTape() as g:
        g.watch(it)
        model = tf.keras.models.Model(input_layer, outputs=x)
        output = model(it)

    dy_dx = g.gradient(output, it)

    print(dy_dx)
    print(output)

    bin = sp.binom(n, k)

    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    epsilon = 1e12
    axs.plot(output[0, :, 0], label='y = PascalRNN(x)')
    axs.plot(dy_dx[0, :, 0], label='dy/dx')
    axs.plot(bin + epsilon, label='$\\binom{\,n}{\,k}$')
    # axs.plot(bin + epsilon, label='${n}\choose{k}$')

    axs.set_xlabel('t')
    for pos in ['right', 'left', 'bottom', 'top']:
        axs.spines[pos].set_visible(False)

    plt.legend()
    plot_filename = f'../experiments/pascal.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()
