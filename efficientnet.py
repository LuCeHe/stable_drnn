import argparse, os, time, json, shutil, socket, random
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from GenericTools.keras_tools.esoteric_callbacks import LearningRateLogger, TimeStopping
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.utils import NumpyEncoder
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0
from neural_models.activations_tf import activations_with_temperature, critical_cws, critical_cbs
# from neural_models.modified_efficientnet import EfficientNetB0

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

EXPERIMENT = os.path.join(EXPERIMENTS, time_string + random_string + '_lsc-effnet')
os.makedirs(EXPERIMENT, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--epochs", default=3, type=int, help="Batch size")
    parser.add_argument("--steps_per_epoch", default=1, type=int, help="Batch size")
    parser.add_argument("--lr", default=.001, type=float, help="Learning rate")
    parser.add_argument("--batch_normalization", default=1, type=int, help="Batch normalization")
    parser.add_argument("--comments", default='', type=str, help="String to activate extra behaviors")
    parser.add_argument("--dataset", default='mnist', type=str, help="Dataset to train on",
                        choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument(
        "--activation", default='cguderman.1', type=str, help="Activation to train on",
        choices=['cguderman1', 'cguderman.1', 'cguderman.01', 'cswish1', 'cswish.1', 'cswish.01', 'relu', 'gelu_new']
    )
    parser.add_argument(
        "--initialization", default='default', type=str, help="Activation to train on",
        choices=['he', 'critical', 'default']
    )
    parser.add_argument("--stop_time", type=int, default=42, help="Seconds assigned to this job")

    args = parser.parse_args()

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

    # model parameters initialization
    activation = activations_with_temperature[args.activation]
    kernel_initializer, bias_initializer = 'default', 'default'
    effnet = EfficientNetB0(
        include_top=False, weights=None, activation=activation,
        batch_normalization=bool(args.batch_normalization),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        comments=args.comments
    )

    readout = tf.keras.layers.Conv2D(
            classes, 7,
            kernel_initializer='he_normal' if kernel_initializer == 'default' else kernel_initializer,
            bias_initializer='zeros' if bias_initializer == 'default' else bias_initializer
        )
    reshape = tf.keras.layers.Reshape((classes, ))


    # model graph construction
    input_layer = tf.keras.layers.Input(input_shape)
    outeff = effnet(input_layer)
    outmodel = reshape(readout(outeff))

    model = tf.keras.models.Model(input_layer, outmodel)
    model.summary()

    # training
    history_path = os.path.join(EXPERIMENT, 'log.csv')

    callbacks = [
        TimeStopping(args.stop_time, 1),
        tf.keras.callbacks.CSVLogger(history_path),
    ]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer, loss, metrics=['sparse_categorical_accuracy', 'sparse_categorical_crossentropy'])
    model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
        callbacks=callbacks, steps_per_epoch=args.steps_per_epoch if args.steps_per_epoch > 0 else None
    )

    results = {}
    evaluation = model.evaluate(x_test, y_test, return_dict=True, verbose=True)
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
    time_start = time.perf_counter()
    results = main()
    time_elapsed = (time.perf_counter() - time_start)
    print('All done, in ' + str(time_elapsed) + 's')

    results.update(time_elapsed=time_elapsed)
    results.update(hostname=socket.gethostname())

    string_result = json.dumps(results, indent=4, cls=NumpyEncoder)
    print(string_result)
    path = os.path.join(EXPERIMENT, 'results.txt')
    with open(path, "w") as f:
        f.write(string_result)

    shutil.make_archive(EXPERIMENT, 'zip', EXPERIMENT)
