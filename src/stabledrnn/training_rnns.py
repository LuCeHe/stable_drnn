

import os, shutil, logging, json, copy, time
import pandas as pd

from pyaromatics.keras_tools.silence_tensorflow import silence_tf

silence_tf()

import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

tf.compat.v1.enable_eager_execution()

from pyaromatics.keras_tools.esoteric_tasks import language_tasks
from pyaromatics.keras_tools.esoteric_callbacks.gradient_tensorboard import ExtendedTensorBoard
from pyaromatics.keras_tools.esoteric_initializers import esoteric_initializers_list, get_initializer
from pyaromatics.keras_tools.esoteric_callbacks import *
from pyaromatics.keras_tools.plot_tools import plot_history
from pyaromatics.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from pyaromatics.stay_organized.utils import setReproducible, str2val, NumpyEncoder, save_results
from pyaromatics.keras_tools.esoteric_callbacks.several_validations import MultipleValidationSets
from pyaromatics.keras_tools.esoteric_tasks.time_task_redirection import Task

from stablespike.neural_models.full_model import build_model
from stablespike.config.config import default_config

try:
    from stable_drnn.neural_models.recLSC import apply_LSC
    from stable_drnn.neural_models.lruLSC import lruLSC, lruLSCffn
except:
    import sys

    sys.path.append('..')

    from stabledrnn.neural_models.recLSC import apply_LSC
    from stabledrnn.neural_models.lruLSC import lruLSC, lruLSCffn


FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-als', base_dir=CDIR, seed=11)
logger = logging.getLogger('stable_drnn')


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 43

    # task and net
    # ps_mnist heidelberg s_mnist
    # wordptb sl_mnist lra_listops lra_text
    task = 'wordptb'

    # test configuration
    epochs = 2
    steps_per_epoch = 2
    batch_size = 8

    # net
    # maLSNN cLSTM LSTM maLSNNb GRU indrnn LMU ssimplernn rsimplernn reslru lru reslruffn
    net = 'maLSNN'
    # zero_mean_isotropic zero_mean learned positional normal onehot zero_mean_normal
    stack = 1
    n_neurons = 3

    embedding = 'learned:None:None:{}'.format(n_neurons) if task in language_tasks else False
    # comments = 'allns_36_nogradreset_dropout:0'
    comments = 'lscdepth:1_36_embproj_nogradreset_dropout:0_findLSC_radius_pretrained_tsteps:2_test'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test_onlypretrain_pretrained_lsclr:0.0001_nbs:16_tsteps:10'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test_onlypretrain_pretrained_lsclr:0.0001_nbs:16_targetnorm:.5'
    comments = 'allns_36_dropout:.0_embproj_mlminputs'

    # optimizer properties
    lr = None  # 7e-4 None
    optimizer_name = 'SWAAdaBeliefLA'  # AdaBelief AdamW SWAAdaBelief AdaBeliefLA
    # optimizer_name = 'Adam'  # AdaBelief AdamW SWAAdaBelief AdaBeliefLA
    lr_schedule = ''  # 'warmup_cosine_restarts'
    weight_decay_prop_lr = None
    weight_decay = 0. if not 'lra_' in task else 0.1
    clipnorm = None  # not 1., to avoid NaN in the embedding, only ptb though

    loss_name = 'sparse_categorical_crossentropy'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    continue_training = ''
    save_model = False

    # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
    stop_time = 21600


@ex.capture
@ex.automain
def main(epochs, steps_per_epoch, batch_size, GPU, task, comments,
         seed, net, n_neurons, lr, stack, loss_name, embedding, optimizer_name,
         lr_schedule, weight_decay, clipnorm, initializer, stop_time, _log):
    time_start = time.perf_counter()

    task_name = task
    net_name = net

    if 'clipping' in comments:
        clipnorm = 1.

    # comments += '_dampf:.5_v0m'
    comments += '_dampf:.5'

    ostack = stack
    stack, batch_size, embedding, n_neurons, lr = default_config(
        stack, batch_size, embedding, n_neurons, lr, task_name, net_name, setting='LSC'
    )

    exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
    comments += '_**folder:' + exp_dir + '**_'

    images_dir = os.path.join(exp_dir, 'images')
    other_dir = os.path.join(exp_dir, 'other_outputs')
    models_dir = os.path.join(exp_dir, 'trained_models')
    # full_mean, full_var = checkTaskMeanVariance(task_name)
    # comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

    ChooseGPU(GPU)
    setReproducible(seed)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    shutil.copytree(os.path.join(CDIR, 'neural_models'), other_dir + '/neural_models_tf')
    shutil.copyfile(FILENAME, other_dir + '/' + os.path.split(FILENAME)[-1])

    timerepeat = str2val(comments, 'timerepeat', int, default=1)
    maxlen = str2val(comments, 'maxlen', int, default=100)
    comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)

    # task definition
    if 'onlypretrain' in comments:
        epochs = 0
        steps_per_epoch = 0
    train_task_args = dict(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                           name=task_name, train_val_test='train', maxlen=maxlen, comments=comments)
    gen_train = Task(**train_task_args)

    comments += '_batchsize:' + str(gen_train.batch_size)

    final_epochs = gen_train.epochs
    final_steps_per_epoch = gen_train.steps_per_epoch

    if initializer in esoteric_initializers_list:
        initializer = get_initializer(initializer_name=initializer)

    comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')
    initial_state = None
    model_args = dict(
        task_name=task_name, net_name=net_name, n_neurons=n_neurons,
        lr=lr, stack=stack, loss_name=loss_name,
        embedding=embedding, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
        weight_decay=weight_decay, clipnorm=clipnorm, initializer=initializer, comments=comments,
        in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len, vocab_size=gen_train.vocab_size,
        n_out=gen_train.out_dim, final_epochs=gen_train.epochs, final_steps_per_epoch=gen_train.steps_per_epoch,
        seed=seed,
    )

    results = {}

    history_path = other_dir + '/log.csv'
    print_every = 2  # int(final_epochs / 10) if not final_epochs < 10 else 1

    if 'findLSC' in comments or 'onlyloadpretrained' in comments:
        print('Finding the LSC...')
        n_samples = str2val(comments, 'normsamples', int, default=-1)
        lscrec = bool(str2val(comments, 'lscrec', int, default=0))
        lscdepth = bool(str2val(comments, 'lscdepth', int, default=0))
        lscout = bool(str2val(comments, 'lscout', int, default=0))
        lscin = bool(str2val(comments, 'lscin', int, default=0))
        naswot = str2val(comments, 'naswot', int, default=0)

        if 'allns' in comments:
            lscrec, lscdepth, lscout, lscin = True, True, True, True

        # n_samples = 100
        norm_pow = str2val(comments, 'normpow', float, default=2)
        norm_pow = norm_pow if norm_pow > 0 else np.inf
        new_model_args = copy.deepcopy(model_args)
        new_comments = new_model_args['comments'] + '_reoldspike'

        # if not 'ssimplernn' in net:
        new_comments = new_comments + '_wmultiplier'
        new_comments = new_comments + '_wshuff'

        new_batch_size = batch_size

        lsclr = 7.2e-3
        # 3.14e-3 # 7.2e-4 # 7.2e-3 #

        if 'ptb' in task_name:

            new_batch_size = 8 if not 'maLSNN' in net_name else 3
            if 'simplernn' in net_name:
                new_batch_size = 16

            new_batch_size = str2val(comments, 'nbs', int, default=new_batch_size)
            new_comments = str2val(new_comments, 'batchsize', replace=new_batch_size)

        if 'heidelberg' in task_name and 'maLSNN' in net_name:
            new_batch_size = 100
            if stack in [7, 5, 3]:
                new_batch_size = 32

            new_comments = str2val(new_comments, 'batchsize', replace=new_batch_size)

        new_model_args['comments'] = new_comments
        new_task_args = copy.deepcopy(train_task_args)
        new_task_args['batch_size'] = new_batch_size

        lscw_filepath = os.path.join(models_dir, 'lsc')
        save_weights_path = lscw_filepath if 'savelscweights' in comments else None

        time_steps = str2val(comments, 'tsteps', int, default=2) if 'test' in comments else None
        print(json.dumps(new_model_args, indent=4, cls=NumpyEncoder))
        lsclr = str2val(comments, 'lsclr', float, default=lsclr)

        if 'reslru' in net_name and 'ffnlsc' in comments:
            weights, lsc_results = lruLSCffn(
                comments=comments, seed=seed, stack=stack, width=n_neurons, classes=gen_train.out_dim,
                vocab_size=gen_train.vocab_size, maxlen=4
            )

        elif 'reslru' in net_name:
            weights, lsc_results = lruLSC(
                comments=comments, seed=seed, stack=stack, width=n_neurons, classes=gen_train.out_dim,
                vocab_size=gen_train.vocab_size, maxlen=gen_train.out_len
            )

        else:

            del gen_train
            weights, lsc_results = apply_LSC(
                train_task_args=new_task_args, model_args=new_model_args, norm_pow=norm_pow, n_samples=n_samples,
                batch_size=new_batch_size,
                rec_norm=lscrec, depth_norm=lscdepth, decoder_norm=lscout, encoder_norm=lscin,
                save_weights_path=save_weights_path, time_steps=time_steps, lr=lsclr, naswot=naswot,
                stop_time=stop_time
            )
        results.update(lsc_results)
        results['lsclr'] = lsclr

        time_lsc = (time.perf_counter() - time_start)
        print('LSC took {} seconds'.format(time_lsc))
        stop_time = int(stop_time - time_lsc)

    gen_train = Task(**train_task_args)
    gen_val = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                   name=task_name, train_val_test='val', maxlen=maxlen, comments=comments)
    gen_test = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                    name=task_name, train_val_test='test', maxlen=maxlen, comments=comments)

    checkpoint_filepath = os.path.join(models_dir, 'checkpoint')

    callbacks = [
        LearningRateLogger(),
        TimeStopping(stop_time, 1),
        # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min',
            save_best_only=True
        ),
    ]
    if not 'lru' in net_name and not 'lra_' in task_name:
        callbacks.append(
            MultipleValidationSets({'v': gen_val, 't': gen_test}, verbose=0)
        )

    callbacks.extend([
        tf.keras.callbacks.CSVLogger(history_path),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ])

    if ostack in [3, 5, 6, 7]:
        callbacks.append(
            ClearMemory(end_of_batch=True, verbose=0, show_gpu=False),
        )

    if 'tenb' in comments:
        val_data = gen_val.__getitem__()
        callbacks.append(
            ExtendedTensorBoard(validation_data=val_data, log_dir=other_dir, histogram_freq=print_every),
        )

    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()

    train_model = build_model(**model_args)
    train_model.summary()

    if 'findLSC' in comments:
        train_model.set_weights(weights)

    results['n_params'] = train_model.count_params()
    save_results(other_dir, results)
    train_model.fit(gen_train, validation_data=gen_val,
                    epochs=final_epochs, steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks)

    actual_epochs = 0
    if final_epochs > 0:
        train_model.load_weights(checkpoint_filepath)

        history_df = pd.read_csv(history_path)

        actual_epochs = history_df['epoch'].iloc[-1] + 1
        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        # plot_filename = os.path.join(images_dir, 'history.png')
        # plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)
        json_filename = os.path.join(other_dir, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        # plot only validation curves
        history_keys = history_df.columns.tolist()
        lengh_keys = 6
        no_vals_keys = [k for k in history_keys if not k.startswith('val_')]
        all_chunks = [no_vals_keys[x:x + lengh_keys] for x in range(0, len(no_vals_keys), lengh_keys)]
        for i, subkeys in enumerate(all_chunks):
            history_dict = {k: history_df[k].tolist() for k in subkeys}
            history_dict.update(
                {'val_' + k: history_df['val_' + k].tolist() for k in subkeys if 'val_' + k in history_keys}
            )
            plot_filename = os.path.join(images_dir, f'history_{i}.png')
            plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)

        removable_checkpoints = sorted([d for d in os.listdir(models_dir) if 'checkpoint' in d])
        for d in removable_checkpoints: os.remove(os.path.join(models_dir, d))

        # results['convergence'] = convergence_estimation(history_dict['val_loss'])

    print('Fitting done!')
    gens = {'train': gen_train, 'val': gen_val, 'test': gen_test}

    for set, gen in gens.items():
        try:
            evaluation = train_model.evaluate(gen_test, return_dict=True, verbose=True)
            for k in evaluation.keys():
                results[set + '_' + k] = evaluation[k]
        except Exception as e:
            print(e)
            print('Evaluation failed: ', set)
            results[set + '_failed'] = str(e)

    results['full_comments'] = comments
    results['final_epochs'] = str(actual_epochs)
    results['final_steps_per_epoch'] = final_steps_per_epoch
    time_elapsed = (time.perf_counter() - time_start)

    results.update(time_elapsed=time_elapsed)

    save_results(other_dir, results)

    print('All done, in ' + str(time_elapsed) + 's')
