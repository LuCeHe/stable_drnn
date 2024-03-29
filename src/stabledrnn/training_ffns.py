import sys

sys.path.append('..')

import argparse, os, time, json, shutil, socket, random, pickle
import numpy as np
import matplotlib.pyplot as plt
from pyaromatics.keras_tools.silence_tensorflow import silence_tf

silence_tf()

import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd

from sklearn.model_selection import train_test_split

from pyaromatics.keras_tools.esoteric_callbacks import TimeStopping
from pyaromatics.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from pyaromatics.keras_tools.plot_tools import plot_history
from pyaromatics.stay_organized.utils import NumpyEncoder, str2val

from stable_drnn.neural_models.nonrecLSC import apply_LSC_no_time

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_lsc-ffnandcnns')
os.makedirs(EXPERIMENT, exist_ok=True)


def get_argparse():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--epochs", default=1, type=int, help="Training Epochs")
    parser.add_argument("--pretrain_epochs", default=2, type=int, help="Pretraining Epochs")  # 20
    parser.add_argument("--steps_per_epoch", default=2, type=int, help="Batch size")  # -1
    parser.add_argument("--layers", default=30, type=int, help="Number of layers")
    parser.add_argument("--resize", default=32, type=int, help="Resize images", choices=[224, 128, 64, 32])
    parser.add_argument("--width", default=128, type=int, help="Layer width")
    parser.add_argument("--lr", default=.001, type=float, help="Learning rate")
    # parser.add_argument("--comments", default='findLSC_supnpsd', type=str, help="String to activate extra behaviors")
    parser.add_argument("--comments", default='findLSC_supsubnpsd_adabelief_pretrained_onlyloadpretrained', type=str,
                        help="String to activate extra behaviors")
    parser.add_argument("--dataset", default='mnist', type=str,
                        choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument("--net_type", default='ffn', type=str, choices=['ffn', 'cnn'])
    parser.add_argument("--activation", default='cos', type=str, help="Activation",
                        choices=['swish', 'relu', 'sin', 'cos'])
    parser.add_argument(
        "--initialization", default='glorot_normal', type=str, help="Activation to train on",
        choices=['he_normal', 'glorot_normal']
    )
    parser.add_argument("--stop_time", type=int, default=3600, help="Seconds assigned to this job")

    args = parser.parse_args()
    return args


def build_cnn(args, input_shape, classes):
    img_input = tf.keras.layers.Input(shape=input_shape)

    # Build stem
    x = img_input
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255.)(x)
    x = tf.keras.layers.experimental.preprocessing.Normalization(mean=.5, variance=1 / 12)(x)

    # 224, 224
    x = tf.keras.layers.experimental.preprocessing.Resizing(args.resize, args.resize, interpolation="bilinear")(x)

    for _ in range(args.layers):
        x = tf.keras.layers.Conv2D(
            args.width, 3,
            # strides=2,
            padding='valid',
            kernel_initializer=args.initialization)(x)
        x = tf.keras.layers.Activation(args.activation)(x)

    x = tf.keras.layers.Conv2D(
        1, 3,
        # strides=2,
        padding='valid',
        kernel_initializer=args.initialization)(x)

    x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
    x = tf.keras.layers.Activation(args.activation)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2 * classes, kernel_initializer=args.initialization)(x)
    x = tf.keras.layers.Activation(args.activation)(x)
    x = tf.keras.layers.Dense(classes, kernel_initializer=args.initialization)(x)

    model = tf.keras.models.Model(img_input, x)
    return model


def build_ffn(args, input_shape, classes):
    img_input = tf.keras.layers.Input(shape=input_shape)

    # Build stem
    x = img_input

    for _ in range(args.layers):
        x = tf.keras.layers.Dense(args.width, kernel_initializer=args.initialization)(x)
        x = tf.keras.layers.Activation(args.activation)(x)

    x = tf.keras.layers.Dense(classes, kernel_initializer=args.initialization)(x)

    model = tf.keras.models.Model(img_input, x)
    return model


def plot_model_weights_dists(model, plot_dir, plot_tag):
    # get model weights
    weights = model.get_weights()

    # weights names
    weights_names = [weight.name for layer in model.layers for weight in layer.weights]

    # plot histograms of all weights in the same plot, in different subplots
    fig, axs = plt.subplots(int(len(weights) / 10) + 1, 10, figsize=(20, 5),
                            gridspec_kw={'hspace': 0.9, 'wspace': 0.3})

    for c, (n, w) in enumerate(zip(weights_names, weights)):
        i, j = c // 10, c % 10
        axs[i, j].hist(w.flatten(), bins=100, label=n)
        axs[i, j].set_title(n)

    # figure title
    fig.suptitle(f'Weights distribution of {plot_tag} model')
    fig.savefig(os.path.join(plot_dir, f'histograms_{plot_tag}.png'))


def main(args):
    results = {}
    results.update(vars(args))

    if 'heinit' in args.comments:
        args.initialization = 'he_normal'

    act_name = args.activation
    if args.activation == 'sin':
        args.activation = tf.math.sin

    elif args.activation == 'cos':
        args.activation = tf.math.cos

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(json.dumps(results, indent=4, cls=NumpyEncoder))

    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, args.dataset).load_data()

    if args.net_type == 'effnet':
        # make mnist grayscale -> rgb
        x_train = x_train if x_train.shape[-1] == 3 else \
            tf.image.resize(x_train[..., None], [32, 32]).numpy().repeat(3, -1)
        x_test = x_test if x_test.shape[-1] == 3 else tf.image.resize(x_test[..., None], [32, 32]).numpy().repeat(3, -1)

    elif args.net_type == 'ffn':
        # flatten data
        n_samples_train = x_train.shape[0]
        n_samples_test = x_test.shape[0]
        x_train = np.reshape(x_train, (n_samples_train, -1))
        x_test = np.reshape(x_test, (n_samples_test, -1))

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    # normalize data
    m, s = x_train.mean(), x_train.std()
    x_train, x_val, x_test = (x_train - m) / s, (x_val - m) / s, (x_test - m) / s

    classes = np.max(y_train) + 1
    input_shape = x_train.shape[1:]

    if args.net_type == 'cnn':
        bm = lambda: build_cnn(args, input_shape, classes)
    elif args.net_type == 'ffn':
        bm = lambda: build_ffn(args, input_shape, classes)
    else:
        raise NotImplementedError

    if 'findLSC' in args.comments or 'onlyloadpretrained' in args.comments:
        model = bm()
        plot_model_weights_dists(model, EXPERIMENT, plot_tag='before')

        del model
        tf.keras.backend.clear_session()

        gen_val = NumpyClassificationGenerator(
            x_train, y_train,
            epochs=args.pretrain_epochs, steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            output_type='[i]o'
        )

        max_dim = str2val(args.comments, 'maxdim', int, default=200000)
        fanin = str2val(args.comments, 'fanin', bool, default=False)
        flsc = str2val(args.comments, 'flsc', bool, default=False)

        comments = args.comments

        comments = comments + '_wmultiplier'
        comments = comments + '_wshuff'
        # comments = comments + '_nosgd'
        custom_objects = {'cos': tf.math.cos, 'sin': tf.math.sin}
        # custom_objects = None
        weights, lsc_results = apply_LSC_no_time(
            bm, generator=gen_val, max_dim=max_dim, norm_pow=2, forward_lsc=flsc,
            nlayerjump=2, net_name='ffn', task_name=args.dataset, activation=act_name, seed=args.seed,
            learning_rate=1e-4,
            comments=comments, stop_time=args.stop_time, custom_objects=custom_objects,
        )

        model = bm()
        model.set_weights(weights)
        plot_model_weights_dists(model, EXPERIMENT, plot_tag='after')

        results.update(lsc_results)
    else:
        model = bm()
    model.summary()

    # training
    history_path = os.path.join(EXPERIMENT, 'log.csv')

    callbacks = [
        TimeStopping(args.stop_time, 1),
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr = args.lr
    if args.lr < 0:
        if act_name == 'cos':
            lr = 1e-4
        elif act_name == 'relu':
            lr = 3.16e-4
        elif act_name == 'sin':
            lr = 1e-4
        else:
            raise NotImplementedError

    if 'adabelief' in args.comments:
        adabelief = tfa.optimizers.AdaBelief(lr=lr, weight_decay=1e-4)
        optimizer = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss, metrics=['sparse_categorical_accuracy', 'sparse_categorical_crossentropy'])
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else None
    epochs = args.epochs if args.epochs > 0 else None

    if 'onlypretrain' in args.comments:
        epochs = 0
        steps_per_epoch = 0

    model.fit(
        x_train, y_train, epochs=epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
        callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )

    if epochs > 0:
        evaluation = model.evaluate(x_test, y_test, return_dict=True, verbose=True, steps=steps_per_epoch,
                                    batch_size=args.batch_size)
        for k in evaluation.keys():
            results['test_' + k] = evaluation[k]

    if epochs > 0:
        history_df = pd.read_csv(history_path)

        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        plot_filename = os.path.join(EXPERIMENT, 'history.png')
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=args.epochs)
        json_filename = os.path.join(EXPERIMENT, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

    return results


if __name__ == "__main__":
    args = get_argparse()

    with open(os.path.join(EXPERIMENT, 'results.txt'), "w") as f:
        f.write(json.dumps(vars(args), indent=4, cls=NumpyEncoder))

    time_start = time.perf_counter()
    results = main(args)
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in ' + str(time_elapsed) + 's')

    results.update(time_elapsed=time_elapsed)
    results.update(hostname=socket.gethostname())

    print(results)
    try:
        with open(os.path.join(EXPERIMENT, 'results.pkl'), 'wb') as fp:
            pickle.dump(results, fp)
    except Exception as e:
        print('Error saving dictionary to pickle file', e)

    try:
        string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
        print(string_result)
        with open(os.path.join(EXPERIMENT, 'results.txt'), "w") as f:
            f.write(string_result)
    except Exception as e:
        print('Error saving dictionary to json file', e)

    shutil.make_archive(EXPERIMENT, 'zip', EXPERIMENT)
