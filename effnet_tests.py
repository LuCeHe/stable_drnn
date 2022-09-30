import tensorflow as tf
import numpy as np

from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.expose_latent import expose_latent_model
from alif_sg.neural_models.sgdLSC import get_norms
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0

batch_size = 4
steps_per_epoch = 3
n_tries = 5
n_exceptions = 0
max_dim = 32
test_weight_id = 10

optimizer = tf.keras.optimizers.Adam(learning_rate=1e2)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# make mnist grayscale -> rgb
x_train = x_train if x_train.shape[-1] == 3 else tf.image.resize(x_train[..., None], [32, 32]).numpy().repeat(3, -1)
x_test = x_test if x_test.shape[-1] == 3 else tf.image.resize(x_test[..., None], [32, 32]).numpy().repeat(3, -1)

classes = np.max(y_train) + 1
input_shape = x_train.shape[1:]

batch = x_train[:batch_size]
batch = tf.convert_to_tensor(tf.cast(batch, tf.float32), dtype=tf.float32)


def sample_axis(tensor, max_dim=1024, return_deshuffling=False):
    # FIXME, not sure if functional for axis different from 1
    axis = 1
    if tensor.shape[axis] > max_dim:
        newdim_inp = sorted(np.random.choice(tensor.shape[axis], max_dim, replace=False))
        tp = tf.transpose(tensor)
        g = tf.gather(tp, indices=newdim_inp)
        out_tensor = tf.transpose(g)
    else:
        out_tensor = tensor

    if not return_deshuffling:
        return out_tensor

    else:
        remaining_indices = list(set(range(tensor.shape[axis])).difference(set(newdim_inp)))

        shuffled_indices = newdim_inp + remaining_indices
        deshuffle_indices = np.array(shuffled_indices).argsort()

        # sample = tf.gather(params, indices=newdim_inp).numpy()
        remainder_transpose = tf.gather(tp, indices=remaining_indices)
        remainder = tf.transpose(remainder_transpose)

        return out_tensor, remainder, deshuffle_indices


def desample_axis(sample, remainder, deshuffle_indices):
    # FIXME, not sure if functional for axis different from 1
    axis = 1

    concat = tf.concat([sample, remainder], axis=axis)
    concat = tf.transpose(concat)
    deshuffled = tf.gather(concat, indices=deshuffle_indices)
    deshuffled = tf.transpose(deshuffled)

    return deshuffled


def split_effnet(almost_seq_model, pairs):
    lnames = [layer.name for layer in almost_seq_model.layers]

    input_shape = almost_seq_model.layers[pairs[0] + 1].input_shape[1:]
    premodel = tf.keras.models.Model(almost_seq_model.inputs, almost_seq_model.get_layer(lnames[pairs[0]]).output)

    DL_input = tf.keras.layers.Input(input_shape)
    DL_model = DL_input
    for layer in almost_seq_model.layers[pairs[0] + 1:pairs[1] + 1]:
        if isinstance(layer.input, list):
            break
        DL_model = layer(DL_model)
    intermodel = tf.keras.models.Model(inputs=DL_input, outputs=DL_model)

    print(pairs)

    return premodel, intermodel


all_norms = []
all_losses = []
weights = None
for i in range(n_tries):
    print('-===-' * 30)
    print('Step', i)

    with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
        tape.watch(batch)

        # split model in three

        effnet = EfficientNetB0()

        if not weights is None:
            effnet.set_weights(weights)

        lnames = [layer.name for layer in effnet.layers]
        for _ in range(3):
            pairs = sorted(np.random.choice(len(lnames), 2, replace=False))
            input_shape = effnet.layers[pairs[0] + 1].input_shape[1:]
            if not isinstance(input_shape, list):
                break

        print('Effnet mean weights', tf.reduce_mean([tf.reduce_mean(tf.cast(t, tf.float32)) for t in effnet.weights]))

        premodel, intermodel = split_effnet(effnet, pairs)

        preinter = premodel(batch)

        tape.watch(preinter)

        # flatten and deflatten to make sure the flat version is in the gradient graph
        # preinter_shape = preinter.shape
        flat_inp = tf.reshape(preinter, [preinter.shape[0], -1])

        # sample and desample to make sure that the sample is in the graph,
        # so the derivative will be taken correctly
        shuffinp, reminder, indices = sample_axis(flat_inp, max_dim=max_dim, return_deshuffling=True)
        defhuffledinp = desample_axis(shuffinp, reminder, indices)

        assert tf.math.reduce_all(tf.equal(defhuffledinp, flat_inp))

        new_preinter = tf.reshape(defhuffledinp, preinter.shape)

        assert tf.math.reduce_all(tf.equal(new_preinter, preinter))

        interout = intermodel(new_preinter)

        inp = shuffinp
        oup = interout

        # inp = tf.reshape(inp, [inp.shape[0], -1])
        oup = tf.reshape(oup, [oup.shape[0], -1])
        oup = sample_axis(oup, max_dim=max_dim)

        norms = get_norms(tape, [inp], [oup], n_samples=100, norm_pow=2)
        loss = tf.reduce_mean(tf.abs(norms - 1))

        all_norms.append(tf.reduce_mean(norms))
        all_losses.append(tf.reduce_mean(loss))

        print(tf.reduce_mean(norms), tf.reduce_mean(loss))

    grads = tape.gradient(loss, intermodel.trainable_weights)
    optimizer.apply_gradients(zip(grads, intermodel.trainable_weights))

    weights = effnet.get_weights()

    print('Effnet mean weights', tf.reduce_mean([tf.reduce_mean(tf.cast(t, tf.float32)) for t in effnet.weights]))
