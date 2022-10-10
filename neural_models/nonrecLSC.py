import tensorflow as tf
import numpy as np
from tqdm import tqdm

from GenericTools.keras_tools.convenience_operations import sample_axis, desample_axis
from GenericTools.keras_tools.esoteric_losses import well_loss
from GenericTools.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from GenericTools.keras_tools.expose_latent import expose_latent_model, split_model
from alif_sg.neural_models.recLSC import get_norms
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0


def apply_LSC_no_time(build_model, generator, max_dim=1024, n_samples=100, norm_pow=2):
    assert callable(build_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e1)

    all_norms = []
    all_losses = []
    weights = None

    ma_loss, ma_norm = None, None
    show_loss, show_norm, show_avw = None, None, None
    n_failures = 0

    for epoch in range(generator.epochs):
        pbar = tqdm(total=generator.steps_per_epoch)

        generator.on_epoch_end()
        for i in range(generator.steps_per_epoch):
            try:
                batch = generator.__getitem__(i)[0]
                batch = tf.convert_to_tensor(tf.cast(batch, tf.float32), dtype=tf.float32)

                with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                    tape.watch(batch)

                    model = build_model()
                    if not weights is None:
                        model.set_weights(weights)

                    lnames = [layer.name for layer in model.layers]

                    for _ in range(3):
                        pairs = sorted(np.random.choice(len(lnames), 2, replace=False))
                        input_shape = model.layers[pairs[0] + 1].input_shape[1:]
                        if not isinstance(input_shape, list):
                            break

                    premodel, intermodel = split_model(model, pairs)

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
                    # print(inp.shape, oup.shape)
                    norms = get_norms(tape, [inp], [oup], n_samples=n_samples, norm_pow=norm_pow)
                    loss = tf.reduce_mean(tf.abs(norms - 1))

                    av_weights = tf.reduce_mean([tf.reduce_mean(tf.cast(t, tf.float32)) for t in model.weights])
                    ma_loss = loss if ma_loss is None else ma_loss * 9 / 10 + loss / 10
                    norm = tf.reduce_mean(norms)
                    ma_norm = norm if ma_norm is None else ma_norm * 9 / 10 + norm / 10
                    all_norms.append(norm)
                    all_losses.append(loss)

                grads = tape.gradient(loss, intermodel.trainable_weights)
                optimizer.apply_gradients(zip(grads, intermodel.trainable_weights))

                weights = model.get_weights()

                show_loss = str(ma_loss.numpy().round(6))
                show_norm = str(ma_norm.numpy().round(3))
                show_avw = str(av_weights.numpy().round(3))

            except Exception as e:
                print(e)
                n_failures += 1

            pbar.update(1)
            pbar.set_description(
                f"Pretrain epoch {epoch + 1} step {i + 1}, "
                f"Loss {show_loss}, Norms {show_norm}, "
                f"Av. Weights {show_avw},"
                f" Fail rate {n_failures / (epoch + 1) / (i + 1)}"
            )

    fail_rate = n_failures / generator.epochs / generator.steps_per_epoch

    del model, generator

    tf.keras.backend.clear_session()
    return weights, all_losses, all_norms, fail_rate


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # make mnist grayscale -> rgb
    x_train = x_train if x_train.shape[-1] == 3 else tf.image.resize(x_train[..., None], [32, 32]).numpy().repeat(3, -1)
    x_test = x_test if x_test.shape[-1] == 3 else tf.image.resize(x_test[..., None], [32, 32]).numpy().repeat(3, -1)

    gen_val = NumpyClassificationGenerator(
        x_train, y_train,
        epochs=3, steps_per_epoch=4,
        batch_size=2,
        output_type='[i]o'
    )
    build_model = lambda: EfficientNetB0()
    apply_LSC_no_time(build_model, generator=gen_val, max_dim=1024, n_samples=100, norm_pow=2)
