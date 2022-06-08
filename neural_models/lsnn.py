import tensorflow_probability as tfp

from GenericTools.keras_tools.esoteric_layers.surrogated_step import *
from GenericTools.stay_organized.utils import str2val

tfd = tfp.distributions


class baseLSNN(tf.keras.layers.Layer):
    """
    LSNN
    """

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(self.init_args.items()))

    def __init__(self, num_neurons=None, tau=20., beta=1.8, tau_adaptation=20,
                 dampening=.3, sharpness=1., ref_period=-1, thr=1., inh_exc=1.,
                 n_regular=0, internal_current=0, initializer='orthogonal',
                 config='', v_eq=0,
                 **kwargs):
        super().__init__(**kwargs)

        ref_period = str2val(config, 'refp', float, default=ref_period)
        dampening = str2val(config, 'dampening', float, default=dampening)
        sharpness = str2val(config, 'sharpness', float, default=sharpness)
        self.init_args = dict(
            num_neurons=num_neurons, tau=tau, tau_adaptation=tau_adaptation,
            ref_period=ref_period, n_regular=n_regular, thr=thr,
            dampening=dampening, sharpness=sharpness, beta=beta, inh_exc=inh_exc, v_eq=v_eq,
            internal_current=internal_current, initializer=initializer, config=config)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons, num_neurons, num_neurons, num_neurons)
        self.mask = tf.ones((self.num_neurons, self.num_neurons)) - tf.eye(self.num_neurons)

    def decay_v(self):
        return tf.exp(-1 / self.tau)

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_rec = self.num_neurons

        input_init = self.initializer
        recurrent_init = self.initializer

        self.input_weights = self.add_weight(shape=(n_in, n_rec), initializer=input_init, name='in_weights')
        self.recurrent_weights = self.add_weight(shape=(n_rec, n_rec), initializer=recurrent_init, name='rec_weights')

        self._beta = tf.concat([tf.zeros(self.n_regular), tf.ones(n_rec - self.n_regular) * self.beta], axis=0)
        self.built = True

        self.spike_type = SurrogatedStep(config=self.config, dampening=self.dampening, sharpness=self.sharpness)

    def recurrent_transform(self):
        return self.mask * self.recurrent_weights

    def refract(self, z, last_spike_distance):
        new_last_spike_distance = last_spike_distance
        if self.ref_period > 0:
            spike_locations = tf.cast(tf.math.not_equal(z, 0), tf.float32)
            non_refractory_neurons = tf.cast(last_spike_distance >= self.ref_period, tf.float32)
            z = non_refractory_neurons * z
            new_last_spike_distance = (last_spike_distance + 1) * (1 - spike_locations)
        return z, new_last_spike_distance

    def spike(self, new_v, thr, *args):
        v_sc = (new_v - thr) / thr if 'vt/t' in self.config else (new_v - thr)

        z = self.spike_type(v_sc)
        return z, v_sc

    def currents_composition(self, inputs, old_spike):

        external_current = inputs @ self.input_weights

        i_in = external_current \
               + old_spike @ self.recurrent_transform() \
               + self.internal_current

        return i_in

    def threshold_dynamic(self, old_a, old_z):
        decay_a = tf.exp(-1 / self.tau_adaptation)

        if not 'noalif' in self.config:
            new_a = decay_a * old_a + (1 - decay_a) * old_z
            athr = self.thr + new_a * self._beta

        else:
            # new_a = decay_a * old_a
            new_a = old_a
            athr = self.thr + old_a * 0

        return athr, new_a

    def voltage_dynamic(self, old_v, i_in, old_z):

        i_reset = - self.athr * old_z

        if 'nogradreset':
            i_reset = tf.stop_gradient(i_reset)

        decay_v = self.decay_v()
        new_v = decay_v * old_v + i_in + i_reset

        return new_v

    def call(self, inputs, states, training=None):
        if not training is None:
            tf.keras.backend.set_learning_phase(training)

        old_z = states[0]
        old_v = states[1]
        old_a = states[2]
        last_spike_distance = states[3]

        i_in = self.currents_composition(inputs, old_z)

        self.athr, new_a = self.threshold_dynamic(old_a, old_z)
        new_v = self.voltage_dynamic(old_v, i_in, old_z)

        z, v_sc = self.spike(new_v, self.athr, last_spike_distance, old_v, old_a, new_a)

        # refractoriness
        z, new_last_spike_distance = self.refract(z, last_spike_distance)
        output, new_state = self.prepare_outputs(new_v, old_v, z, old_z, new_a, old_a, new_last_spike_distance,
                                                 last_spike_distance, self.athr, v_sc)
        return output, new_state

    def prepare_outputs(self, new_v, old_v, z, old_z, new_a, old_a, new_last_spike_distance, last_spike_distance, thr,
                        v_sc):

        output = (z, new_v, thr, v_sc)
        new_state = (z, new_v, new_a, new_last_spike_distance)
        return output, new_state


LSNN = baseLSNN


class aLSNN(baseLSNN):
    """
    LSNN where all parameters can be learned
    """

    def build(self, input_shape):
        n_input = input_shape[-1]

        parameter2trainable = {k: v for k, v in self.__dict__.items()
                               if k in ['tau_adaptation', 'thr', 'beta', 'tau']}

        for k, p in parameter2trainable.items():
            initializer = tf.keras.initializers.TruncatedNormal(mean=p, stddev=3 * p / 7)
            initializer = tf.keras.initializers.Constant(value=p)
            p = self.add_weight(shape=(self.num_neurons,), initializer=initializer, name=k, trainable=True)
            self.__dict__.update({k: p})

        if 'LSC' in self.config:
            # initializer = tf.keras.initializers.TruncatedNormal(mean=1, stddev=3 * 1 / 7)
            # initializer = tf.keras.initializers.Constant(value=1)
            # print(tf.maximum(initializer), tf.minimum(initializer))
            # decay_a = tf.exp(-1 / self.tau_adaptation)
            tau = -1 / tf.math.log(1 / 3)
            # tau = -1 / tf.math.log(.86)
            print(tau)
            self.tau = self.add_weight(shape=(self.num_neurons,), initializer=tf.keras.initializers.Constant(value=tau),
                                       name='tau', trainable=True)

        super().build(input_shape)

        self._beta = self.beta
