import argparse, os, time, json, shutil, socket, random, copy
import numpy as np

from pyaromatics.keras_tools.esoteric_activations.smoothrelus import Guderman_T, Swish_T
from pyaromatics.keras_tools.esoteric_regularizers import GeneralActivityRegularization
from pyaromatics.keras_tools.silence_tensorflow import silence_tf
from alif_sg.tools.config import default_eff_lr

silence_tf()

import tensorflow as tf
import tensorflow_addons as tfa

import pandas as pd

from sklearn.model_selection import train_test_split

from pyaromatics.keras_tools.esoteric_callbacks import LearningRateLogger, TimeStopping
from pyaromatics.keras_tools.esoteric_tasks.numpy_generator import NumpyClassificationGenerator
from pyaromatics.keras_tools.plot_tools import plot_history
from pyaromatics.stay_organized.utils import NumpyEncoder, str2val
from alif_sg.neural_models.modified_efficientnet import EfficientNetB0, act_reg

from alif_sg.neural_models.nonrecLSC import apply_LSC_no_time

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

GEXPERIMENTS = os.path.abspath(os.path.join(CDIR, 'good_experiments'))
os.makedirs(GEXPERIMENTS, exist_ok=True)

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
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--epochs", default=3, type=int, help="Batch size")
    parser.add_argument("--steps_per_epoch", default=3, type=int, help="Batch size")
    parser.add_argument("--lr", default=-1, type=float, help="Learning rate")
    parser.add_argument("--batch_normalization", default=1, type=int, help="Batch normalization")
    parser.add_argument("--comments",
                        default='newarch_lscvar',
                        # default='newarch',
                        # default='newarch_pretrained_deslice_findLSC_onlyprem_preprocessinput_meanaxis',
                        # default='newarch_pretrained_deslice_findLSC_truersplit_preprocessinput',
                        type=str, help="String to activate extra behaviors")
    parser.add_argument("--dataset", default='cifar100', type=str, help="Dataset to train on",
                        choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument("--activation", default='tanh', type=str, help="Activation",
                        choices=['swish', 'relu', 'gudermanlu', 'swish.1', 'gudermanlu.1', 'tanh'])
    parser.add_argument("--initialization", default='default', type=str, help="Activation to train on",
                        choices=['he', 'critical', 'default'])
    parser.add_argument("--stop_time", type=int, default=42, help="Seconds assigned to this job")

    args = parser.parse_args()
    return args


def build_model(args, input_shape, classes):
    # model parameters initialization
    kernel_initializer, bias_initializer = 'default', 'default'
    activation = args.activation if not args.activation in extra_acts.keys() else extra_acts[args.activation]

    input_layer = tf.keras.layers.Input(input_shape)

    outmodel = EfficientNetB0(
        include_top=False, weights=None, activation=activation,
        batch_normalization=bool(args.batch_normalization),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        comments=args.comments,
        input_tensor=input_layer,
    )

    if not 'noresize' in args.comments:
        readout = tf.keras.layers.Conv2D(4, 3)
        reshape = tf.keras.layers.Reshape((classes,))
        outmodel = reshape(readout(outmodel))

    model = tf.keras.models.Model(input_layer, outmodel)
    return model


def main(args):
    if not 'preprocessinput' in args.comments:
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
            batch_size=2,
            output_type='[i]o'
        )

        lsc_args = copy.deepcopy(args)
        # lsc_args.comments = lsc_args.comments.replace('findLSC', 'findLSC_noresize')

        bm = lambda: build_model(lsc_args, input_shape, classes)
        maxdim = str2val(args.comments, 'maxdim', int, default=1024)
        lsclr = str2val(args.comments, 'lsclr', float, default=1.0e-3)

        subsample_axis = True if not 'deslice' in args.comments else False

        weights, lsc_results = apply_LSC_no_time(
            build_model=bm, generator=gen_train, max_dim=maxdim, norm_pow=2, comments=args.comments,
            net_name='eff', seed=args.seed, task_name=args.dataset, activation=args.activation,
            subsample_axis=subsample_axis,
            learning_rate=lsclr,
            skip_in_layers=['rescaling', 'normalization', 'resizing'],
            skip_out_layers=['rescaling', 'normalization', 'resizing'],
        )
        model = bm()
        model.set_weights(weights)

        results.update(lsc_results)
        results.update(lsclr=lsclr)

    elif 'lscvar' in args.comments:
        model = build_model(args, input_shape, classes)
        loss = lambda x, y: 0
        lsclr = str2val(args.comments, 'lsclr', float, default=1.0e-4)
        adabelief = tfa.optimizers.AdaBelief(lr=lsclr, weight_decay=1e-4)
        optimizer = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)
        model.compile(optimizer, loss)
        steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else None

        path_pretrained = os.path.join(
            GEXPERIMENTS, f"pretrained_s{args.seed}_effnet_{args.dataset}_{args.activation}_lscvar.h5"
        )

        if os.path.exists(path_pretrained):
            print('Loading pretrained lsc weights')
            model = tf.keras.models.load_model(
                path_pretrained,
                custom_objects={
                    'GeneralActivityRegularization': GeneralActivityRegularization, 'act_reg': act_reg,
                }
            )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(path_pretrained, monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
        ]

        if not 'onlyloadpretrained' in args.comments:
            lsc_history = model.fit(
                x_train, y_train, epochs=10, batch_size=args.batch_size, validation_data=(x_val, y_val),
                callbacks=callbacks, steps_per_epoch=steps_per_epoch
            )
            results.update(LSC_losses=np.array(lsc_history.history['val_loss']) / 18,
                           LSC_norms=np.array(lsc_history.history['val_loss']) / 18 + 1)
            print(lsc_history.history)

        else:

            evaluation = model.evaluate(x_val, y_val, return_dict=True, verbose=True, steps=steps_per_epoch,
                                        batch_size=args.batch_size)
            results.update(LSC_losses=np.array(evaluation['val_loss']) / 18,
                           LSC_norms=np.array(evaluation['val_loss']) / 18 + 1)

        print(args.comments)
        args.comments = args.comments.replace('_lscvar', '')
        print(args.comments)
        weights = model.get_weights()
        model = build_model(args, input_shape, classes)
        model.set_weights(weights)

    else:
        model = build_model(args, input_shape, classes)
    model.summary()

    # training
    history_path = os.path.join(EXPERIMENT, 'log.csv')

    callbacks = [
        TimeStopping(args.stop_time, 1),
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]
    lr = default_eff_lr(args.activation, args.lr, args.batch_normalization)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer, loss, metrics=['sparse_categorical_accuracy', 'sparse_categorical_crossentropy'])
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else None
    epochs = args.epochs

    if 'onlypretrain' in args.comments:
        epochs = 0
        steps_per_epoch = 0

    model.fit(
        x_train, y_train, epochs=epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
        callbacks=callbacks, steps_per_epoch=steps_per_epoch
    )

    try:
        evaluation = model.evaluate(x_test, y_test, return_dict=True, verbose=True, steps=steps_per_epoch,
                                    batch_size=args.batch_size)
        for k in evaluation.keys():
            results['test_' + k] = evaluation[k]
    except:
        pass

    if epochs > 0:
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
