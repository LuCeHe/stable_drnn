import tensorflow as tf


class PascalRNN(tf.keras.layers.Layer):

    def __init__(self, num_neurons=None, **kwargs):
        super().__init__(**kwargs)

        self.init_args = dict(num_neurons=num_neurons)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons,)

    def call(self, inputs, states, **kwargs):
        output = inputs + states
        new_state = output
        return output, new_state



if __name__ == '__main__':

    L = 100
    # input_layer= tf.keras.layers.InputLayer(input_shape=[None, 1])
    layers = [PascalRNN(num_neurons=1) for _ in range(L)]

    model = tf.keras.Sequential(layers)

    it = tf.random.uniform(shape=[1, 100])
