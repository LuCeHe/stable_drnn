import os, shutil, logging, json

import numpy as np
import pandas as pd

import tensorflow as tf

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
from GenericTools.stay_organized.utils import timeStructured, setReproducible, str2val

from alif_sg.generate_data.task_redirection import Task, checkTaskMeanVariance
# from alif_sg.visualization_tools.training_tests import Tests
from alif_sg.neural_models.full_model import build_model

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-als', base_dir=CDIR, seed=11)
logger = logging.getLogger('alif_sg')

language_tasks = ['ptb', 'wiki103', 'wmt14', 'time_ae_merge', 'monkey', 'wordptb']


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 41

    # task and net
    # ptb time_ae simplest_random time_ae_merge ps_mnist heidelberg wiki103 wmt14 s_mnist xor small_s_mnist
    # wordptb sl_mnist
    task_name = 'sl_mnist'

    # test configuration
    epochs = 3
    steps_per_epoch = 1
    batch_size = 32
    stack = 2

    # net
    # mn_aLSNN_2 mn_aLSNN_2_sig LSNN maLSNN spikingPerformer smallGPT2 aLSNN_noIC spikingLSTM
    net_name = 'aLSNN'
    # zero_mean_isotropic zero_mean learned positional normal onehot zero_mean_normal
    n_neurons = None
    embedding = 'learned:None:None:{}'.format(n_neurons) if task_name in language_tasks else False

    comments = 'dampening:.3'

    # optimizer properties
    lr = None  # 7e-4
    optimizer_name = 'SGD'  # AdaBelief AdamW SWAAdaBelief
    lr_schedule = ''  # 'warmup_cosine_restarts'
    weight_decay_prop_lr = None
    weight_decay = .01 if not 'mnist' in task_name else 0.  # weight_decay_prop_lr * lr
    clipnorm = None  # not 1., to avoid NaN in the embedding, only ptb though

    loss_name = 'sparse_categorical_crossentropy'  # categorical_crossentropy categorical_focal_loss contrastive_loss
    initializer = 'he_uniform'  # uniform glorot_uniform orthogonal glorot_normal NoZeroGlorot

    continue_training = ''
    save_model = False

    # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
    stop_time = 21600


task_max_epochs = {
    'ptb': np.inf,  # 100,
    'ps_mnist': np.inf,
    's_mnist': np.inf,
    'heidelberg': np.inf,
    'wmt14': np.inf,
    'monkey': np.inf,
}


@ex.capture
@ex.automain
def main(epochs, steps_per_epoch, batch_size, GPU, task_name, comments,
         seed, net_name, n_neurons, lr, stack, loss_name, embedding, optimizer_name,
         lr_schedule, weight_decay, clipnorm, initializer, stop_time, _log):

    if n_neurons is None:
        if task_name in language_tasks:
            n_neurons = 1300
        elif task_name == 'heidelberg':
            n_neurons = 256
        elif 'mnist' in task_name:
            n_neurons = 128
        else:
            raise NotImplementedError

    exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
    comments += '_**folder:' + exp_dir + '**_'

    config_dir = os.path.join(exp_dir, '1')
    images_dir = os.path.join(exp_dir, 'images')
    other_dir = os.path.join(exp_dir, 'other_outputs')
    models_dir = os.path.join(exp_dir, 'trained_models')
    full_mean, full_var = checkTaskMeanVariance(task_name)
    comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

    ChooseGPU(GPU)
    setReproducible(seed)

    shutil.copytree(os.path.join(CDIR, 'neural_models'), other_dir + '/neural_models')
    # shutil.copyfile(os.path.join(CDIR, 'run_tf2.sh'), other_dir + '/run_tf2.sh')
    shutil.copyfile(FILENAME, other_dir + '/' + os.path.split(FILENAME)[-1])

    timerepeat = str2val(comments, 'timerepeat', int, default=1)
    maxlen = str2val(comments, 'maxlen', int, default=100)
    comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)

    # task definition
    gen_train = Task(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                     name=task_name, train_val_test='train', maxlen=maxlen, comments=comments, lr=lr)
    gen_val = Task(timerepeat=timerepeat, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                   name=task_name, train_val_test='val', maxlen=maxlen, comments=comments, lr=lr)

    comments += '_batchsize:' + str(gen_train.batch_size)

    final_epochs = gen_train.epochs
    final_steps_per_epoch = gen_train.steps_per_epoch
    tau_adaptation = str2val(comments, 'taub', float, default=int(gen_train.in_len / 2))
    tau = str2val(comments, 'tauv', float, default=.1)
    # tau_adaptation = int(gen_train.in_len / 2)  # 200 800 4000

    if initializer in esoteric_initializers_list:
        initializer = get_initializer(initializer_name=initializer)

    comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')
    train_model = build_model(
        task_name=task_name, net_name=net_name, n_neurons=n_neurons, tau=tau, n_dt_per_step=timerepeat,
        neutral_phase_length=0, lr=gen_train.lr, batch_size=gen_train.batch_size, stack=stack, loss_name=loss_name,
        embedding=embedding, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
        weight_decay=weight_decay, clipnorm=clipnorm, initializer=initializer, comments=comments,
        language_tasks=language_tasks,
        in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
        n_out=gen_train.out_dim, tau_adaptation=tau_adaptation,
        final_epochs=gen_train.epochs, final_steps_per_epoch=gen_train.steps_per_epoch)

    results = {}

    train_model.summary()

    history_path = other_dir + '/log.csv'
    print_every = 2  # int(final_epochs / 10) if not final_epochs < 10 else 1
    val_data = gen_val.__getitem__()

    checkpoint_filepath = os.path.join(models_dir, 'checkpoint')
    callbacks = [
        LearningRateLogger(),
        VariablesLogger(variables_to_log=['hard_heaviside']),
        tf.keras.callbacks.CSVLogger(history_path),
        TimeStopping(stop_time, 1),  # 22h=79200 s, 21h=75600 s, 20h=72000 s, 12h = 43200 s, 6h = 21600 s, 72h = 259200
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True
        )
    ]


    if 'tenb' in comments:
        callbacks.append(
            ExtendedTensorBoard(validation_data=val_data, log_dir=other_dir, histogram_freq=print_every),
        )

    # plots before training
    # Tests(task_name, gen_test, train_model, images_dir, save_pickle=False, subdir_name='nontrained')

    train_model.fit(gen_train, validation_data=gen_val,
                    epochs=final_epochs, steps_per_epoch=steps_per_epoch,
                    callbacks=callbacks)

    actual_epochs = 0
    if final_epochs > 0:
        train_model.load_weights(checkpoint_filepath)
        history_df = pd.read_csv(history_path)

        actual_epochs = history_df['epoch'].iloc[-1] + 1
        # results['accumulated_epochs'] = str(int(results['accumulated_epochs']) + int(actual_epochs))
        history_dict = {k: history_df[k].tolist() for k in history_df.columns.tolist()}

        plot_filename = os.path.join(*[images_dir, 'history.png'])
        plot_history(histories=history_dict, plot_filename=plot_filename, epochs=final_epochs)
        json_filename = os.path.join(*[other_dir, 'history.json'])
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

    # plots after training
    stateful = True if 'ptb' in task_name else False
    if 'stateful' in comments: stateful = True
    timerepeat = timerepeat if not 'repetitionsschedule' in comments else timerepeat - 1
    for tr in [timerepeat, 1, 5, 10]:
        try:
            bs = 8 if not stateful else batch_size
            gen_test = Task(timerepeat=tr, batch_size=bs, steps_per_epoch=steps_per_epoch,
                            name=task_name, train_val_test='test', maxlen=maxlen, comments=comments)

            # if tr == 1:
            #     Tests(task_name, gen_test, train_model, images_dir, save_pickle=False)

            evaluation = train_model.evaluate(gen_test, return_dict=True, verbose=True)
            for k in evaluation.keys():
                results[k + '_test_{}'.format(tr)] = evaluation[k]

        except Exception as e:
            print('Error while testing: ', e)

    # batch = gen_val.__getitem__()
    # results['score_JS'] = IWPJS(train_model, batch)
    results['n_params'] = train_model.count_params()
    results['final_epochs'] = str(actual_epochs)
    results['final_steps_per_epoch'] = final_steps_per_epoch

    results_filename = os.path.join(*[other_dir, 'results.json'])
    json.dump(results, open(results_filename, "w"))
