import os, shutil, logging, json, copy

import pandas as pd

from GenericTools.keras_tools.silence_tensorflow import silence_tf

silence_tf()

import tensorflow as tf
from sg_design_lif.neural_models.config import default_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

tf.compat.v1.enable_eager_execution()

from GenericTools.keras_tools.convergence_metric import convergence_estimation
from GenericTools.keras_tools.esoteric_callbacks.gradient_tensorboard import ExtendedTensorBoard
from GenericTools.keras_tools.esoteric_initializers import esoteric_initializers_list, get_initializer
from GenericTools.keras_tools.esoteric_callbacks import *
from GenericTools.keras_tools.plot_tools import plot_history
from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.stay_organized.utils import timeStructured, setReproducible, str2val, NumpyEncoder
from GenericTools.keras_tools.esoteric_callbacks.several_validations import MultipleValidationSets
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task, checkTaskMeanVariance, language_tasks

from sg_design_lif.neural_models.full_model import build_model
from alif_sg.neural_models.recLSC import apply_LSC

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-als', base_dir=CDIR, seed=11)
logger = logging.getLogger('alif_sg')


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 41

    # task and net
    # ps_mnist heidelberg s_mnist
    # wordptb sl_mnist
    task_name = 'heidelberg'

    # test configuration
    epochs = 0
    steps_per_epoch = 1
    batch_size = 2
    stack = None

    # net
    # maLSNN cLSTM LSTM
    net_name = 'GRU'
    # zero_mean_isotropic zero_mean learned positional normal onehot zero_mean_normal
    n_neurons = None

    embedding = 'learned:None:None:{}'.format(n_neurons) if task_name in language_tasks else False

    comments = 'findLSC_test_lscdepth:1_lscout:1_gaussbeta_test_gausslsc_ptb1'  # 'nsLIFreadout_adaptsg_dropout:0.50' findLSC_test
    comments = ''  # 'nsLIFreadout_adaptsg_dropout:0.50' findLSC_test

    # optimizer properties
    lr = None  # 7e-4 None
    optimizer_name = 'AdamW'  # AdaBelief AdamW SWAAdaBelief
    lr_schedule = ''  # 'warmup_cosine_restarts'
    weight_decay_prop_lr = None
    weight_decay = .0 if not 'mnist' in task_name else 0.  # weight_decay_prop_lr * lr
    clipnorm = None  # not 1., to avoid NaN in the embedding, only ptb though

    loss_name = 'sparse_categorical_crossentropy'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'glorot_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    continue_training = ''
    save_model = False

    # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
    stop_time = 21600


@ex.capture
@ex.automain
def main(epochs, steps_per_epoch, batch_size, GPU, task_name, comments,
         seed, net_name, n_neurons, lr, stack, loss_name, embedding, optimizer_name,
         lr_schedule, weight_decay, clipnorm, initializer, stop_time, _log):
    stack, batch_size, embedding, n_neurons, lr = default_config(
        stack, batch_size, embedding, n_neurons, lr, task_name, lsc=True
    )

    if task_name == 'heidelberg':
        sLSTM_factor = .37  # 1 / 3
    elif task_name == 'sl_mnist':
        sLSTM_factor = 1 / 3
    else:
        sLSTM_factor = 1 / 3

    n_neurons = n_neurons if not 'LSTM' in net_name else int(n_neurons * sLSTM_factor)
    stack = '700:300' if ('LSTM' in net_name and task_name == 'wordptb') else stack

    exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
    comments += '_**folder:' + exp_dir + '**_'

    images_dir = os.path.join(exp_dir, 'images')
    other_dir = os.path.join(exp_dir, 'other_outputs')
    models_dir = os.path.join(exp_dir, 'trained_models')
    full_mean, full_var = checkTaskMeanVariance(task_name)
    comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

    ChooseGPU(GPU)
    setReproducible(seed)

    shutil.copytree(os.path.join(CDIR, 'neural_models'), other_dir + '/neural_models')
    shutil.copyfile(FILENAME, other_dir + '/' + os.path.split(FILENAME)[-1])

    timerepeat = str2val(comments, 'timerepeat', int, default=1)
    maxlen = str2val(comments, 'maxlen', int, default=100)
    comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)

    # task definition
    train_task_args = dict(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                           name=task_name, train_val_test='train', maxlen=maxlen, comments=comments)
    gen_train = Task(**train_task_args)
    gen_val = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                   name=task_name, train_val_test='val', maxlen=maxlen, comments=comments)
    gen_test = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                    name=task_name, train_val_test='test', maxlen=maxlen, comments=comments)

    comments += '_batchsize:' + str(gen_train.batch_size)

    final_epochs = gen_train.epochs
    final_steps_per_epoch = gen_train.steps_per_epoch

    if initializer in esoteric_initializers_list:
        initializer = get_initializer(initializer_name=initializer)

    comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')
    # task_name, net_name, n_neurons, tau, lr, stack,
    # loss_name, embedding, optimizer_name, tau_adaptation, lr_schedule, weight_decay, clipnorm,
    # initializer, comments, in_len, n_in, out_len, n_out, final_epochs,
    initial_state = None
    model_args = dict(
        task_name=task_name, net_name=net_name, n_neurons=n_neurons,
        lr=lr, stack=stack, loss_name=loss_name,
        embedding=embedding, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
        weight_decay=weight_decay, clipnorm=clipnorm, initializer=initializer, comments=comments,
        in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
        n_out=gen_train.out_dim, final_epochs=gen_train.epochs,
    )

    results = {}

    history_path = other_dir + '/log.csv'
    print_every = 2  # int(final_epochs / 10) if not final_epochs < 10 else 1
    val_data = gen_val.__getitem__()

    checkpoint_filepath = os.path.join(models_dir, 'checkpoint')
    callbacks = [
        LearningRateLogger(),
        TimeStopping(stop_time, 1),  # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True
        ),
        MultipleValidationSets({'v': gen_val, 't': gen_test}, verbose=0),
        tf.keras.callbacks.CSVLogger(history_path),
    ]

    if 'tenb' in comments:
        callbacks.append(
            ExtendedTensorBoard(validation_data=val_data, log_dir=other_dir, histogram_freq=print_every),
        )

    if 'findLSC' in comments:
        n_samples = str2val(comments, 'normsamples', int, default=None)
        lscdepth = bool(str2val(comments, 'lscdepth', int, default=0))
        lscout = bool(str2val(comments, 'lscout', int, default=0))
        if n_samples is None:
            n_samples = 100 if not 'ptb' in task_name else 10

        # n_samples = 100
        norm_pow = str2val(comments, 'normpow', float, default=2)
        norm_pow = norm_pow if norm_pow > 0 else np.inf
        new_model_args = copy.deepcopy(model_args)
        new_model_args['comments'] = new_model_args['comments'] + '_reoldspike'

        new_task_args = copy.deepcopy(train_task_args)
        new_task_args['batch_size'] = new_task_args['batch_size'] if not 'ptb' in task_name else 8
        weights, lsc_results = apply_LSC(
            train_task_args=train_task_args, model_args=new_model_args, norm_pow=norm_pow, n_samples=n_samples,
            batch_size=batch_size, depth_norm=lscdepth, decoder_norm=lscout
        )
        results.update(lsc_results)

    train_model = build_model(**model_args)
    train_model.summary()

    if 'findLSC' in comments:
        train_model.set_weights(weights)

    train_model.fit(gen_train, validation_data=gen_val,
                    epochs=final_epochs, steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks)

    actual_epochs = 0
    if final_epochs > 0:
        train_model.load_weights(checkpoint_filepath)

        history_df = pd.read_csv(history_path)

        actual_epochs = history_df['epoch'].iloc[-1] + 1
        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        plot_filename = os.path.join(images_dir, 'history.png')
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)
        json_filename = os.path.join(other_dir, 'history.json')
        history_jsonable = {k: np.array(v).astype(float).tolist() for k, v in history_dict.items()}
        json.dump(history_jsonable, open(json_filename, "w"))

        # plot only validation curves
        history_dict = {k: history_df[k].tolist() if 'val' in k else [] for k in history_df.columns.tolist()}
        plot_filename = os.path.join(images_dir, 'history_val.png')
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)

        removable_checkpoints = sorted([d for d in os.listdir(models_dir) if 'checkpoint' in d])
        for d in removable_checkpoints: os.remove(os.path.join(models_dir, d))

        results['convergence'] = convergence_estimation(history_dict['val_loss'])

    print('Fitting done!')

    evaluation = train_model.evaluate(gen_test, return_dict=True, verbose=True)
    for k in evaluation.keys():
        results['test_' + k] = evaluation[k]

    gen_v = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                 name=task_name, train_val_test='val', maxlen=maxlen, comments=comments)

    evaluation = train_model.evaluate(gen_v, return_dict=True, verbose=True)
    for k in evaluation.keys():
        results['valval_' + k] = evaluation[k]

    results['n_params'] = train_model.count_params()
    results['full_comments'] = comments
    results['final_epochs'] = str(actual_epochs)
    results['final_steps_per_epoch'] = final_steps_per_epoch

    results_filename = os.path.join(other_dir, 'results.json')
    json.dump(results, open(results_filename, "w"))
    print('DONE')
