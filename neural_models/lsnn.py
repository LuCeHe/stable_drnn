import tensorflow_probability as tfp

from GenericTools.keras_tools.esoteric_initializers import FuncOnInitializer
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
                 ref_period=-1, thr=1., n_regular=0, internal_current=0, initializer='orthogonal',
                 config='', v_eq=0,
                 **kwargs):
        super().__init__(**kwargs)

        ref_period = str2val(config, 'refp', float, default=ref_period)
        dampening = str2val(config, 'dampening', float, default=.3)
        sharpness = str2val(config, 'sharpness', float, default=1.)
        self.init_args = dict(
            num_neurons=num_neurons, tau=tau, tau_adaptation=tau_adaptation,
            ref_period=ref_period, n_regular=n_regular, thr=thr,
            dampening=dampening, sharpness=sharpness, beta=beta, v_eq=v_eq,
            internal_current=internal_current, initializer=initializer, config=config)
        self.__dict__.update(self.init_args)

        self.state_size = (num_neurons, num_neurons, num_neurons, num_neurons)
        self.mask = tf.ones((self.num_neurons, self.num_neurons)) - tf.eye(self.num_neurons)

    def decay_v(self):
        return tf.exp(-1 / self.tau)

    def build(self, input_shape):
        n_in = input_shape[-1]
        n_rec = self.num_neurons

        self.input_weights = self.add_weight(shape=(n_in, n_rec), initializer=self.initializer, name='in_weights')
        self.recurrent_weights = self.add_weight(shape=(n_rec, n_rec), initializer=self.initializer, name='rec_weights')

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


class aLSNN(baseLSNN):
    """
    LSNN where all parameters can be learned
    """

    def build(self, input_shape):
        n_input = input_shape[-1]

        parameter2trainable = {k: v for k, v in self.__dict__.items()
                               if k in ['tau_adaptation', 'thr', 'beta', 'tau']}

        for k, p in parameter2trainable.items():
            # initializer = tf.keras.initializers.TruncatedNormal(mean=p, stddev=3 * p / 7)
            initializer = tf.keras.initializers.Constant(value=p)
            p = self.add_weight(shape=(self.num_neurons,), initializer=initializer, name=k, trainable=True)
            self.__dict__.update({k: p})

        super().build(input_shape)

        if 'LSC' in self.config:
            alpha_v = .92  # 1/3 .86
            tau = -1 / tf.math.log(alpha_v)
            # print(tau)
            self.tau = self.add_weight(shape=(self.num_neurons,), initializer=tf.keras.initializers.Constant(value=tau),
                                       name='tau', trainable=True)

            alpha_a = .92  # 1/3 .86
            tau_adaptation = -1 / tf.math.log(alpha_a)
            self.tau_adaptation = self.add_weight(shape=(self.num_neurons,),
                                                  initializer=tf.keras.initializers.Constant(value=tau_adaptation),
                                                  name='tau_adaptation', trainable=True)

            abs_var_in = tf.reduce_mean(tf.abs(self.input_weights))
            self.dampening = 1 / abs_var_in / 2 / n_input

            # beta = 1 / self.dampening
            beta = str2val(self.config, 'beta', float, default=1 / self.dampening)
            self.beta = self.add_weight(shape=(self.num_neurons,),
                                        initializer=tf.keras.initializers.Constant(value=beta),
                                        name='beta', trainable=True)

            thr = (1 - alpha_a) / self.dampening
            self.thr = self.add_weight(shape=(self.num_neurons,), initializer=tf.keras.initializers.Constant(value=thr),
                                       name='thr', trainable=True)

            abs_var_rec = tf.reduce_mean(tf.abs(self.recurrent_weights))
            self.recurrent_weights = self.recurrent_weights / abs_var_rec * abs_var_in \
                                     * n_input / (self.num_neurons - 1)

            # print(beta, self.dampening)

        elif 'lsc' in self.config:
            # print('here?')
            alpha_v = .92  # 1/3 .86
            tau = -1 / tf.math.log(alpha_v)
            # print(tau)
            self.tau = self.add_weight(shape=(self.num_neurons,), initializer=tf.keras.initializers.Constant(value=tau),
                                       name='tau', trainable=True)

            alpha_a = .92  # 1/3 .86
            tau_adaptation = -1 / tf.math.log(alpha_a)
            self.tau_adaptation = self.add_weight(shape=(self.num_neurons,),
                                                  initializer=tf.keras.initializers.Constant(value=tau_adaptation),
                                                  name='tau_adaptation', trainable=True)

            abs_var_in = tf.reduce_mean(tf.abs(self.input_weights))
            self.input_weights = self.input_weights / abs_var_in / n_input
            self.dampening = 1 / 2

            # beta = 1 / self.dampening
            beta = str2val(self.config, 'beta', float, default=1 / self.dampening)
            self.beta = self.add_weight(shape=(self.num_neurons,),
                                        initializer=tf.keras.initializers.Constant(value=beta),
                                        name='beta', trainable=True)

            thr = (1 - alpha_a) / self.dampening
            self.thr = self.add_weight(shape=(self.num_neurons,), initializer=tf.keras.initializers.Constant(value=thr),
                                       name='thr', trainable=True)

            abs_var_rec = tf.reduce_mean(tf.abs(self.recurrent_weights))
            self.recurrent_weights = self.recurrent_weights / abs_var_rec / (self.num_neurons - 1)

            # print(beta, self.dampening)

        elif 'randominit' in self.config:
            linv = lambda x: -1 / tf.math.log(x)
            self.tau = self.add_weight(shape=(self.num_neurons,),
                                       initializer=FuncOnInitializer(
                                           linv,
                                           tf.keras.initializers.RandomUniform(minval=0.3, maxval=.99)
                                       ),
                                       name='tau', trainable=True)
            self.tau_adaptation = self.add_weight(shape=(self.num_neurons,),
                                                  initializer=FuncOnInitializer(
                                                      linv,
                                                      tf.keras.initializers.RandomUniform(minval=0.3, maxval=.99)
                                                  ),
                                                  name='tau_adaptation', trainable=True)
            self.dampening = tf.random.uniform((self.num_neurons,), minval=0.2, maxval=1.)
            self.beta = self.add_weight(shape=(self.num_neurons,),
                                        initializer=tf.keras.initializers.RandomUniform(minval=1., maxval=2.),
                                        name='beta', trainable=True)
            self.thr = self.add_weight(shape=(self.num_neurons,),
                                       initializer=tf.keras.initializers.RandomUniform(minval=0.01, maxval=1.),
                                       name='thr', trainable=True)

        else:
            self.tau = self.add_weight(shape=(self.num_neurons,),
                                       initializer=tf.keras.initializers.Constant(value=20),
                                       name='tau', trainable=False)
            self.tau_adaptation = self.add_weight(shape=(self.num_neurons,),
                                                  initializer=tf.keras.initializers.Constant(value=700),
                                                  name='tau_adaptation', trainable=True)
            self.dampening = .3
            self.beta = self.add_weight(shape=(self.num_neurons,),
                                        initializer=tf.keras.initializers.Constant(value=1.8),
                                        name='beta', trainable=True)
            self.thr = self.add_weight(shape=(self.num_neurons,),
                                       initializer=tf.keras.initializers.Constant(value=.1),
                                       name='thr', trainable=True)

        self._beta = self.beta
        dampening = str2val(self.config, 'dampening', float, default=self.dampening)

        self.spike_type = SurrogatedStep(config=self.config, dampening=dampening, sharpness=self.sharpness)


LSNN = baseLSNN
