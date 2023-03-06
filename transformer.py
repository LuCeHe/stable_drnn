from __future__ import (absolute_import, division, print_function, unicode_literals)
import os, argparse, random, socket, time, json, shutil

import tensorflow as tf
from tensorflow_addons.optimizers import AdamW

import numpy as np
import pandas as pd

from GenericTools.keras_tools.esoteric_activations.smoothrelus import Guderman_T, Swish_T
from GenericTools.keras_tools.esoteric_callbacks import LearningRateLogger, TimeStopping, CSVLogger
from GenericTools.keras_tools.esoteric_losses import sparse_perplexity
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.utils import NumpyEncoder, str2val
from alif_sg.neural_models.chunked_transformer import chunked_lsc
from alif_sg.neural_models.nonrecLSC import apply_LSC_no_time
from alif_sg.neural_models.transformer_model import build_model
from filmformer.generation_data.data_loader import WMT_ENDE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
DATADIR = os.path.abspath(os.path.join(CDIR, '..', 'data', 'wmt'))
os.makedirs(DATADIR, exist_ok=True)

EXPERIMENTS = os.path.join(CDIR, 'experiments')
named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y-%m-%d--%H-%M-%S--", named_tuple)
random_string = ''.join([str(r) for r in np.random.choice(10, 4)])

extra_acts = {
    'gudermanlu': Guderman_T(),
    'gudermanlu.1': Guderman_T(.1),
    'swish.1': Swish_T(.1),
}


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments",
                        # default='deslice_findLSC_meanaxis_truersplit',
                        default='chunked_meanaxis_sameemb_noimagloss_normri_findLSC_radius',
                        # default='pretrained_deslice_sameemb_truersplit_findLSC_supsubnpsd',
                        # default='',
                        type=str, help="String to activate extra behaviors")
    parser.add_argument("--activation", default='swish', type=str, help="Network non-linearity")
    parser.add_argument("--seed", default=5, type=int, help="Random seed")
    parser.add_argument("--epochs", default=3, type=int, help="Epochs")
    parser.add_argument("--pretraining_epochs", default=200, type=int, help="Pretraining Epochs")
    parser.add_argument("--steps_per_epoch", default=2, type=int, help="Steps per epoch")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--stop_time", default=60, type=int, help="Stop time")
    parser.add_argument("--results_dir", default=EXPERIMENTS, type=str, help="Experiments Folder")
    parser.add_argument("--lr", default=3.16e-5, type=float, help="Experiments Folder")
    args = parser.parse_args()
    return args


def main(args, experiment_dir):
    comments = args.comments
    results = vars(args)
    print(json.dumps(vars(args), indent=4, cls=NumpyEncoder))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # hyper paramaters
    TRAIN_RATIO = 0.9
    D_MODEL = 512
    D_POINT_WISE_FF = D_MODEL * 4
    ENCODER_COUNT = DECODER_COUNT = 6
    ATTENTION_HEAD_COUNT = 8
    DROPOUT_PROB = 0.1
    SEQ_MAX_LEN_SOURCE = 100
    SEQ_MAX_LEN_TARGET = 101
    BPE_VOCAB_SIZE = 32000
    pretraining_epochs = args.pretraining_epochs

    # for overfitting test hyper parameters
    # BATCH_SIZE = 32
    # EPOCHS = 100
    DATA_LIMIT = None

    if 'DESKTOP-4MV34QG' in socket.gethostname():
        DATA_LIMIT = 200000
        BATCH_SIZE = 2
        D_MODEL = 8
        ATTENTION_HEAD_COUNT = 2
        pretraining_epochs = 30
        # comments += 'test'

    GLOBAL_BATCH_SIZE = (args.batch_size * 1)

    activation = args.activation if not args.activation in extra_acts.keys() else extra_acts[args.activation]

    bm = lambda: build_model(
        inputs_timesteps=SEQ_MAX_LEN_SOURCE,
        target_timesteps=SEQ_MAX_LEN_TARGET,
        inputs_vocab_size=BPE_VOCAB_SIZE,
        target_vocab_size=BPE_VOCAB_SIZE,
        encoder_count=ENCODER_COUNT,
        decoder_count=DECODER_COUNT,
        attention_head_count=ATTENTION_HEAD_COUNT,
        d_model=D_MODEL,
        d_point_wise_ff=D_POINT_WISE_FF,
        dropout_prob=DROPOUT_PROB,
        batch_size=args.batch_size,
        activation=activation,
        comments=args.comments,
    )

    if 'chunked' in args.comments and 'findLSC' in args.comments:
        weights, lsc_results = chunked_lsc(
            SEQ_MAX_LEN_SOURCE=SEQ_MAX_LEN_SOURCE,
            SEQ_MAX_LEN_TARGET=SEQ_MAX_LEN_TARGET,
            pretrain_SEQ_MAX_LEN_SOURCE=50,
            BPE_VOCAB_SIZE=BPE_VOCAB_SIZE,
            encoder_count=ENCODER_COUNT,
            decoder_count=DECODER_COUNT,
            attention_head_count=ATTENTION_HEAD_COUNT,
            d_model=D_MODEL,
            d_point_wise_ff=D_POINT_WISE_FF,
            dropout_prob=DROPOUT_PROB,
            activation=activation,
            comments=args.comments,
            plot_pretraining=False,
            layer_index=0,
            epochs=pretraining_epochs,
        )
        results.update(lsc_results)
        model = bm()
        model.set_weights(weights)


    elif 'findLSC' in args.comments:
        lsc_batch_size = 8
        bm = lambda: build_model(
            inputs_timesteps=SEQ_MAX_LEN_SOURCE,
            target_timesteps=SEQ_MAX_LEN_TARGET,
            inputs_vocab_size=BPE_VOCAB_SIZE,
            target_vocab_size=BPE_VOCAB_SIZE,
            encoder_count=ENCODER_COUNT,
            decoder_count=DECODER_COUNT,
            attention_head_count=ATTENTION_HEAD_COUNT,
            d_model=D_MODEL,
            d_point_wise_ff=D_POINT_WISE_FF,
            dropout_prob=DROPOUT_PROB,
            batch_size=lsc_batch_size,
            activation=activation,
            comments=args.comments,
        )

        gen_lsc = WMT_ENDE(
            data_dir=DATADIR, batch_size=lsc_batch_size, bpe_vocab_size=BPE_VOCAB_SIZE,
            seq_max_len_source=SEQ_MAX_LEN_SOURCE, seq_max_len_target=SEQ_MAX_LEN_TARGET, data_limit=DATA_LIMIT,
            train_ratio=TRAIN_RATIO, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, data_split='train',
            comments=comments
        )

        max_dim = str2val(args.comments, 'maxdim', int, default=1024)
        lsclr = 1e-3,
        weights, lsc_results = apply_LSC_no_time(
            bm, generator=gen_lsc, max_dim=max_dim, norm_pow=2, nlayerjump=2,
            skip_in_layers=['embeddinglayer', 'dropout', 'de_concatenate'],
            skip_out_layers=['input', 'tf.linalg.matmul', 'dropout', 'embedding'],
            keep_in_layers=['encoder', 'concatenate'],
            # keep_out_layers=['identity_'],
            net_name='trasnf', task_name='ende', seed=args.seed, activation=args.activation,
            learning_rate=lsclr,
            comments=args.comments,
        )

        del gen_lsc
        model = bm()
        model.set_weights(weights)

        results.update(lsc_results)
        results.update(lsclr=lsclr)
    else:
        model = bm()
    model.summary()

    gen_train = WMT_ENDE(
        data_dir=DATADIR, batch_size=GLOBAL_BATCH_SIZE, bpe_vocab_size=BPE_VOCAB_SIZE,
        seq_max_len_source=SEQ_MAX_LEN_SOURCE, seq_max_len_target=SEQ_MAX_LEN_TARGET, data_limit=DATA_LIMIT,
        train_ratio=TRAIN_RATIO, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, data_split='train',
        comments=comments
    )

    gen_val = WMT_ENDE(
        data_dir=DATADIR, batch_size=GLOBAL_BATCH_SIZE, bpe_vocab_size=BPE_VOCAB_SIZE,
        seq_max_len_source=SEQ_MAX_LEN_SOURCE, seq_max_len_target=SEQ_MAX_LEN_TARGET, data_limit=DATA_LIMIT,
        train_ratio=TRAIN_RATIO, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, data_split='val',
        comments=comments
    )

    learning_rate = args.lr
    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[
            'sparse_categorical_accuracy', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            sparse_perplexity
        ],
    )

    history_path = os.path.join(experiment_dir, 'history.csv')
    callbacks = [
        LearningRateLogger(),
        TimeStopping(args.stop_time, 1),
        CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    model.fit(gen_train, validation_data=gen_val, callbacks=callbacks, epochs=args.epochs)

    if args.epochs > 0:
        history_df = pd.read_csv(history_path)

        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}
        json_filename = os.path.join(experiment_dir, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        history_keys = history_df.columns.tolist()
        lengh_keys = 6
        no_vals_keys = [k for k in history_keys if not k.startswith('val_')]
        all_chunks = [no_vals_keys[x:x + lengh_keys] for x in range(0, len(no_vals_keys), lengh_keys)]
        for i, subkeys in enumerate(all_chunks):
            history_dict = {k: history_df[k].tolist() for k in subkeys}
            history_dict.update(
                {'val_' + k: history_df['val_' + k].tolist() for k in subkeys if 'val_' + k in history_keys}
            )
            plot_filename = os.path.join(experiment_dir, f'history_{i}.png')
            plot_history(histories=history_dict, plot_filename=plot_filename, epochs=args.epochs)

    results['n_params'] = model.count_params()
    return results


if __name__ == "__main__":
    args = get_argparse()

    EXPERIMENT = os.path.join(args.results_dir, time_string + random_string + '_lsc-transformer')
    os.makedirs(EXPERIMENT, exist_ok=True)

    string_result = json.dumps(vars(args), indent=4, cls=NumpyEncoder)
    print(string_result)
    path = os.path.join(EXPERIMENT, 'results.txt')
    with open(path, "w") as f:
        f.write(string_result)

    time_start = time.perf_counter()
    results = main(args, EXPERIMENT)
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
