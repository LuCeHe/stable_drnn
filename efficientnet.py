import argparse, os, time, json, shutil, socket, random
import numpy as np

from GenericTools.keras_tools.esoteric_activations.smoothrelus import Guderman_T, Swish_T
from GenericTools.keras_tools.silence_tensorflow import silence_tf
from alif_sg.tools.config import default_eff_lr

silence_tf()

import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from GenericTools.keras_tools.esoteric_callbacks import LearningRateLogger, TimeStopping
from GenericTools.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.utils import NumpyEncoder, str2val
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0

from alif_sg.neural_models.nonrecLSC import apply_LSC_no_time

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_lsc-effnet')
os.makedirs(EXPERIMENT, exist_ok=True)

extra_acts = {
    'gudermanlu': Guderman_T(),
    'gudermanlu.1': Guderman_T(.1),
    'swish.1': Swish_T(.1),
}


def get_argparse():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--epochs", default=3, type=int, help="Batch size")
    parser.add_argument("--steps_per_epoch", default=3, type=int, help="Batch size")
    parser.add_argument("--lr", default=.001, type=float, help="Learning rate")
    parser.add_argument("--batch_normalization", default=1, type=int, help="Batch normalization")
    parser.add_argument("--comments", default='findLSC_pretrained_truersplit', type=str, help="String to activate extra behaviors")
    parser.add_argument("--dataset", default='cifar100', type=str, help="Dataset to train on",
                        choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument("--activation", default='tanh', type=str, help="Activation",
                        choices=['swish', 'relu', 'gudermanlu', 'swish.1', 'gudermanlu.1', 'tanh'])
    parser.add_argument(
        "--initialization", default='default', type=str, help="Activation to train on",
        choices=['he', 'critical', 'default']
    )
    parser.add_argument("--stop_time", type=int, default=42, help="Seconds assigned to this job")

    args = parser.parse_args()
    return args


def build_model(args, input_shape, classes, effnet=None):
    # model parameters initialization
    kernel_initializer, bias_initializer = 'default', 'default'
    activation = args.activation if not args.activation in extra_acts.keys() else extra_acts[args.activation]
    if effnet is None:
        effnet = EfficientNetB0(
            include_top=False, weights=None, activation=activation,
            batch_normalization=bool(args.batch_normalization),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            comments=args.comments
        )

    readout = tf.keras.layers.Conv2D(4, 3)
    reshape = tf.keras.layers.Reshape((classes,))

    # model graph construction
    input_layer = tf.keras.layers.Input(input_shape)
    outeff = effnet(input_layer)

    # outmodel = readout(outeff)
    # print(outmodel.shape)
    outmodel = reshape(readout(outeff))

    model = tf.keras.models.Model(input_layer, outmodel)
    return model


def main(args):
    args.comments = args.comments + '_preprocessinput'

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(json.dumps(vars(args), indent=4, cls=NumpyEncoder))

    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, args.dataset).load_data()

    # make mnist grayscale -> rgb
    x_train = x_train if x_train.shape[-1] == 3 else tf.image.resize(x_train[..., None], [32, 32]).numpy().repeat(3, -1)
    x_test = x_test if x_test.shape[-1] == 3 else tf.image.resize(x_test[..., None], [32, 32]).numpy().repeat(3, -1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    classes = np.max(y_train) + 1
    input_shape = x_train.shape[1:]

    results = {}
    if 'findLSC' in args.comments:
        gen_train = NumpyClassificationGenerator(
            x_train, y_train,
            epochs=3, steps_per_epoch=args.steps_per_epoch,
            batch_size=4,
            output_type='[i]o'
        )
        activation = args.activation if not args.activation in extra_acts.keys() else extra_acts[args.activation]

        kernel_initializer, bias_initializer = 'default', 'default'
        bm = lambda: EfficientNetB0(
            include_top=False, weights=None, activation=activation,
            batch_normalization=bool(args.batch_normalization),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            comments=args.comments
        )
        max_dim = str2val(args.comments, 'maxdim', int, default=64)

        weights, lsc_results = apply_LSC_no_time(
            bm, generator=gen_train, max_dim=max_dim, norm_pow=2, comments=args.comments,
            net_name='eff', seed=args.seed, task_name=args.dataset, activation=args.activation,
            skip_in_layers=['rescaling', 'normalization', 'resizing'],
            skip_out_layers=['rescaling', 'normalization', 'resizing'],
        )
        effnet = bm()
        effnet.set_weights(weights)
        model = build_model(args, input_shape, classes, effnet=effnet)

        results.update(lsc_results)
    else:
        model = build_model(args, input_shape, classes)
    model.summary()

    # training
    history_path = os.path.join(EXPERIMENT, 'log.csv')

    callbacks = [
        TimeStopping(args.stop_time, 1),
        tf.keras.callbacks.CSVLogger(history_path),
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
    lr = default_eff_lr(args.activation, args.lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss, metrics=['sparse_categorical_accuracy', 'sparse_categorical_crossentropy'])
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else None
    model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
        callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )

    evaluation = model.evaluate(x_test, y_test, return_dict=True, verbose=True, steps=steps_per_epoch,
                                batch_size=args.batch_size)
    for k in evaluation.keys():
        results['test_' + k] = evaluation[k]

    if args.epochs > 0:
        history_df = pd.read_csv(history_path)

        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        plot_filename = os.path.join(EXPERIMENT, 'history.png')
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=args.epochs)
        json_filename = os.path.join(EXPERIMENT, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

    results.update(vars(args))

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

    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
    print(string_result)
    with open(os.path.join(EXPERIMENT, 'results.txt'), "w") as f:
        f.write(string_result)

    shutil.make_archive(EXPERIMENT, 'zip', EXPERIMENT)
