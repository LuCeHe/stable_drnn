import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation


class customLSTMcell(tf.keras.layers.Layer):

    def get_config(self):
        return self.init_args

    def __init__(self, num_neurons=None, activation_gates='sigmoid', activation_c='tanh', activation_h='tanh',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 string_config='', **kwargs):
        self.init_args = dict(num_neurons=num_neurons, activation_gates=activation_gates, activation_c=activation_c,
                              activation_h=activation_h, string_config=string_config)
        super().__init__(**kwargs)
        self.__dict__.update(self.init_args)

        self.string_config = string_config

        self.activation_gates = activation_gates
        self.activation_c = activation_c
        self.activation_h = activation_h

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.state_size = (num_neurons, num_neurons)

    def build(self, input_shape):
        n_in = input_shape[-1]

        self.w_f = self.add_weight(shape=(n_in, self.num_neurons,), initializer=self.kernel_initializer,
                                   name='w_f', trainable=True)
        self.u_f = self.add_weight(shape=(self.num_neurons, self.num_neurons),
                                   initializer=self.recurrent_initializer,
                                   name='u_f', trainable=True)
        self.b_f = self.add_weight(shape=(self.num_neurons,), initializer=self.bias_initializer,
                                   name='b_f', trainable=True)

        self.w_o = self.add_weight(shape=(n_in, self.num_neurons,), initializer=self.kernel_initializer,
                                   name='w_o', trainable=True)
        self.u_o = self.add_weight(shape=(self.num_neurons, self.num_neurons),
                                   initializer=self.recurrent_initializer,
                                   name='u_o', trainable=True)
        self.b_o = self.add_weight(shape=(self.num_neurons,), initializer=self.bias_initializer,
                                   name='b_o', trainable=True)

        self.w_i = self.add_weight(shape=(n_in, self.num_neurons,), initializer=self.kernel_initializer,
                                   name='w_i', trainable=True)
        self.u_i = self.add_weight(shape=(self.num_neurons, self.num_neurons),
                                   initializer=self.recurrent_initializer,
                                   name='u_i', trainable=True)
        self.b_i = self.add_weight(shape=(self.num_neurons,), initializer=self.bias_initializer,
                                   name='b_i', trainable=True)

        self.w_c = self.add_weight(shape=(n_in, self.num_neurons,), initializer=self.kernel_initializer,
                                   name='w_c', trainable=True)
        self.u_c = self.add_weight(shape=(self.num_neurons, self.num_neurons),
                                   initializer=self.recurrent_initializer,
                                   name='u_c', trainable=True)
        self.b_c = self.add_weight(shape=(self.num_neurons,), initializer=self.bias_initializer,
                                   name='b_c', trainable=True)

        if 'LSC' in self.string_config:
            var_uo = tf.reduce_mean(tf.abs(self.u_o))
            self.u_o = self.u_o / var_uo * 5 / self.num_neurons

            var_wo = tf.reduce_mean(tf.abs(self.w_o))
            self.w_o = self.w_o / var_wo * 5 / self.num_neurons

            var_wf = tf.reduce_mean(tf.abs(self.w_f))
            self.w_f = self.w_f / var_wf * 8 / self.num_neurons / 3

            var_wi = tf.reduce_mean(tf.abs(self.w_i))
            self.w_i = self.w_i / var_wi * 8 / self.num_neurons / 3

            var_wc = tf.reduce_mean(tf.abs(self.w_c))
            self.w_c = self.w_c / var_wc * 8 / self.num_neurons / 9

            var_uf = tf.reduce_mean(tf.abs(self.u_f))
            self.u_f = self.u_f / var_uf * 4 / self.num_neurons / 3

            var_ui = tf.reduce_mean(tf.abs(self.u_i))
            self.u_i = self.u_i / var_ui * 4 / self.num_neurons / 3

            var_uc = tf.reduce_mean(tf.abs(self.u_c))
            self.u_c = self.u_c / var_uc * 4 / self.num_neurons / 9

        self.built = True

    def call(self, inputs, states, **kwargs):

        old_c, old_h = states
        f = Activation(self.activation_gates)(inputs @ self.w_f + old_h @ self.u_f + self.b_f)
        i = Activation(self.activation_gates)(inputs @ self.w_i + old_h @ self.u_i + self.b_i)
        o = Activation(self.activation_gates)(inputs @ self.w_o + old_h @ self.u_o + self.b_o)
        c_tilde = Activation(self.activation_c)(inputs @ self.w_c + old_h @ self.u_c + self.b_c)

        c = f * old_c + i * c_tilde
        h = o * Activation(self.activation_h)(c)

        output = (h,)
        new_state = (c, h)
        return output, new_state


if __name__ == '__main__':
    comments = 'LSC'
    n_neurons = 2
    time_steps, in_dim, out_dim = 3, 4, 5

    cell = customLSTMcell(n_neurons, string_config=comments)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, stateful=False)

    lin = tf.keras.layers.Input((time_steps, in_dim))
    readout = tf.keras.layers.Dense(out_dim, name='readout')(rnn(lin))
    model = tf.keras.models.Model(lin, readout)

    print(rnn.cell.u_c)
