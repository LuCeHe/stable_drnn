import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.expose_latent import expose_latent_model
from alif_sg.generate_data.task_redirection import Task
from alif_sg.neural_models.full_model import build_model

np.random.seed(42)
tf.random.set_seed(42)

input_dim = 2
time_steps = 3
batch_size = 2
units = 4
norm_pow = 0.1  # np.inf
n_samples = 11
lr = 1e-2

batch = tf.random.normal((batch_size, time_steps, input_dim))
h0 = tf.random.normal((batch_size, units))
c0 = tf.random.normal((batch_size, units))



def build_model_custom(cell_name='lstm'):
    input_layer = tf.keras.Input(shape=(1, input_dim), batch_size=batch_size)
    hi = tf.keras.Input(shape=(units,), batch_size=batch_size)
    ci = tf.keras.Input(shape=(units,), batch_size=batch_size)

    cell = tf.keras.layers.LSTMCell(units)

    lstm = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True, stateful=True)

    lstm_out, hidden_state, cell_state = lstm(input_layer, initial_state=(hi, ci))

    model = tf.keras.Model(inputs=[input_layer, hi, ci], outputs=[lstm_out, hidden_state, cell_state])
    return model


gen_train = Task(timerepeat=2, epochs=2, batch_size=batch_size, steps_per_epoch=2,
             name='heidelberg', train_val_test='train', maxlen=100, comments='')


model_args = dict(task_name='heidelberg', net_name='maLSNN', n_neurons=units, tau=10,
                  lr=lr, stack=2, loss_name='MSE',
                  embedding=None, optimizer_name='AdaBelief', lr_schedule='',
                  weight_decay=1, clipnorm=1, initializer='glorot_uniform', comments='',
                  in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
                  n_out=gen_train.out_dim, tau_adaptation=1,
                  final_epochs=1)
model = build_model(**model_args)

ht, ct = h0, c0

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

e_model = expose_latent_model(model, include_layers=['encoder'], idx=[1, 2])

losses = []
weights = None
for t in range(time_steps):
    print(t, '-' * 30)
    bt = batch[:, t, :][:, None]
    with tf.GradientTape(persistent=True) as tape:
        # tape.watch(model)
        tape.watch(bt)
        tape.watch(ht)
        tape.watch(ct)
        model = build_model()
        if not weights is None:
            model.set_weights(weights)

        otp1, htp1, ctp1 = model([bt, ht, ct], training=True)

        phh = tape.batch_jacobian(htp1, ht)
        pcc = tape.batch_jacobian(ctp1, ct)
        pch = tape.batch_jacobian(ctp1, ht)
        phc = tape.batch_jacobian(htp1, ct)
        # print(phh.shape, pcc.shape, pch.shape, phc.shape)

        # transition derivative
        td_1 = tf.concat([phh, phc], axis=1)
        td_2 = tf.concat([pch, pcc], axis=1)
        td = tf.concat([td_1, td_2], axis=2)

        x = tf.random.normal((td.shape[0], td.shape[-1], n_samples))
        x_norm = tf.norm(x, ord=norm_pow, axis=1)
        e = tf.einsum('bij,bjk->bik', td, x)
        e_norm = tf.norm(e, ord=norm_pow, axis=1)

        norms = e_norm / x_norm
        norms = tf.reduce_max(norms, axis=-1)
        loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    print('Norms:            ', norms)
    print('Mean params: ', [tf.reduce_mean(w) for w in model.trainable_weights])
    print('Loss:             ', loss)
    losses.append(loss)
    ht, ct = htp1, ctp1

    weights = model.get_weights()
    tf.keras.backend.clear_session()
    del model
    # print(grad)
    # print(np.var(grad) * units)

plt.plot(losses)
plt.show()
