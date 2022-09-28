import tensorflow as tf
import numpy as np

from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.expose_latent import expose_latent_model
from alif_sg.neural_models.sgdLSC import get_norms
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0

batch_size = 2
steps_per_epoch = 3
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# make mnist grayscale -> rgb
x_train = x_train if x_train.shape[-1] == 3 else tf.image.resize(x_train[..., None], [32, 32]).numpy().repeat(3, -1)
x_test = x_test if x_test.shape[-1] == 3 else tf.image.resize(x_test[..., None], [32, 32]).numpy().repeat(3, -1)

classes = np.max(y_train) + 1
input_shape = x_train.shape[1:]

readout = tf.keras.layers.Conv2D(classes, 7, kernel_initializer='he_normal', bias_initializer='zeros')
reshape = tf.keras.layers.Reshape((classes,))

# model graph construction
input_layer = tf.keras.layers.Input(input_shape)
# outeff = effnet(input_layer)
# outmodel = reshape(readout(outeff))
#
# model = tf.keras.models.Model(input_layer, outmodel)
# model.summary()

# lnames = [layer.name for layer in effnet.layers]
# neff = expose_latent_model(effnet)

batch = x_train[:batch_size]
batch = tf.convert_to_tensor(tf.cast(batch, tf.float32), dtype=tf.float32)


def sample_axis(tensor, axis=1, max_dim=1024):
    if tensor.shape[axis] > max_dim:
        newdim_inp = sorted(np.random.choice(tensor.shape[axis], max_dim, replace=False))
        tp = tf.transpose(tensor)
        g = tf.gather(tp, indices=newdim_inp)
        tensor = tf.transpose(g)

    return tensor


#
# all_norms = []
# all_losses = []
# for i in range(steps_per_epoch):
#     print('Step', i)
#     with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
#         tape.watch(batch)
#
#         effnet = EfficientNetB0()
#         neff, layer_names = expose_latent_model(effnet, return_names=True)
#         print(layer_names)
#         outputs = neff(batch)
#
#         pairs = sorted(np.random.choice(len(layer_names), 2, replace=False))
#         pairs = [10, -1]
#
#         inp = tf.reshape(outputs[pairs[0]], [outputs[pairs[0]].shape[0], -1])
#         oup = tf.reshape(outputs[pairs[1]], [outputs[pairs[1]].shape[0], -1])
#
#         inp = neff.get_layer(layer_names[pairs[0]])
# inp = sample_axis(inp, axis=1, max_dim=1024)
# oup = sample_axis(oup, axis=1, max_dim=1024)

# j = tape.batch_jacobian(inp, oup)
# print(j.shape)
#     norms = get_norms(tape, [inp], [oup], n_samples=100, norm_pow=2)
#     # norms = get_norms(tape, [batch], [oup], n_samples=100, norm_pow=2)
#
#     loss = well_loss(min_value=1, max_value=1, walls_type='relu', axis='all')(norms)
#     all_norms.append(tf.reduce_mean(norms))
#     all_losses.append(tf.reduce_mean(loss))
#
#     print(pairs)
#     print(tf.reduce_mean(norms), tf.reduce_mean(loss))
#
# grads = tape.gradient(loss, neff.trainable_weights)
# print(grads)
# optimizer.apply_gradients(zip(grads, neff.trainable_weights))


effnet = EfficientNetB0()

lnames = [layer.name for layer in effnet.layers]

# assert all([len(effnet.get_layer(ln).output) == 1 for ln in lnames])

pairs = sorted(np.random.choice(len(lnames), 2, replace=False))
# pairs = [10, -1]
print(pairs)
premodel = tf.keras.models.Model(effnet.inputs, effnet.get_layer(lnames[pairs[0]]).output)
intermodel = tf.keras.models.Model(
    effnet.get_layer(lnames[pairs[0] + 1]).output,
    effnet.get_layer(lnames[pairs[1]]).output
)

preinter = premodel(batch)
interout = intermodel(preinter)
print(interout)
