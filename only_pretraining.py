import os, shutil, logging, json, copy
import pandas as pd

from pyaromatics.keras_tools.silence_tensorflow import silence_tf

silence_tf()

from sg_design_lif.neural_models.config import default_config

import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

tf.compat.v1.enable_eager_execution()

from pyaromatics.keras_tools.convergence_metric import convergence_estimation
from pyaromatics.keras_tools.esoteric_callbacks.gradient_tensorboard import ExtendedTensorBoard
from pyaromatics.keras_tools.esoteric_initializers import esoteric_initializers_list, get_initializer
from pyaromatics.keras_tools.esoteric_callbacks import *
from pyaromatics.keras_tools.plot_tools import plot_history
from pyaromatics.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from pyaromatics.stay_organized.utils import timeStructured, setReproducible, str2val, NumpyEncoder, save_results
from pyaromatics.keras_tools.esoteric_callbacks.several_validations import MultipleValidationSets
from pyaromatics.keras_tools.esoteric_tasks.time_task_redirection import Task, checkTaskMeanVariance, language_tasks

from sg_design_lif.neural_models.full_model import build_model
from alif_sg.neural_models.recLSC import apply_LSC
from alif_sg.neural_models.nonrecLSC import apply_LSC_no_time

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-pretr', base_dir=CDIR, seed=11)
logger = logging.getLogger('alif_sg')


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 43

    # task and net
    # ps_mnist heidelberg s_mnist
    # wordptb sl_mnist
    task = 'sl_mnist'

    # test configuration
    epochs = 4
    steps_per_epoch = 2
    batch_size = 2

    # net
    # maLSNN cLSTM LSTM maLSNNb GRU indrnn LMU ssimplernn rsimplernn
    net = 'maLSNN'
    # zero_mean_isotropic zero_mean learned positional normal onehot zero_mean_normal
    stack = 5
    n_neurons = 2

    embedding = 'learned:None:None:{}'.format(n_neurons) if task in language_tasks else False
    comments = '36_embproj_nogradreset_dropout:.3_timerepeat:2_lscdepth:1_findLSC_supsubnpsd_test_pretrained_deslice'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_lscdepth:1_findLSC_supsubnpsd_test_pretrained'
    # comments = '36_embproj_nogradreset_dropout:.3_timerepeat:2_lscdepth:1_findLSC_supsubnpsd_test_pretrained_randlsc'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_targetnorm:.5_test'
    comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_test_onlypretrain_lscshuffw_gausslsc'
    comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_findLSC_radius_test_onlypretrain_lscshuffw_gausslsc'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test_onlypretrain'
    comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_pretrained'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test_onlypretrain_pretrained_lsclr:0.0001_nbs:16_tsteps:10'
    # comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test_onlypretrain_pretrained_lsclr:0.0001_nbs:16_targetnorm:.5'
    # comments = ''

    # optimizer properties
    lr = None  # 7e-4 None
    optimizer_name = 'AdaBeliefLA'  # AdaBelief AdamW SWAAdaBelief AdaBeliefLA
    lr_schedule = ''  # 'warmup_cosine_restarts'
    weight_decay_prop_lr = None
    weight_decay = 0.  # weight_decay_prop_lr * lr
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
    # create folder

    FILENAME = os.path.realpath(__file__)
    CDIR = os.path.dirname(FILENAME)
    EXPERIMENTS = os.path.join(CDIR, 'experiments')
    gradwincrease = os.path.join(EXPERIMENTS, 'gradualwidthincrease')
    os.makedirs(gradwincrease, exist_ok=True)

    gradwincrease = os.path.join(EXPERIMENTS, f'{net}_{task}')
    os.makedirs(gradwincrease, exist_ok=True)

    for n_neurons in [2, 4, 8, 16, 32]:
        task_name = task
        net_name = net

        comments += '_dampf:.5'

        ostack = stack
        stack, batch_size, embedding, n_neurons, lr = default_config(
            stack, batch_size, embedding, n_neurons, lr, task_name, net_name, setting='LSC'
        )

        exp_dir = os.path.join(CDIR, ex.observers[0].basedir)
        comments += '_**folder:' + exp_dir + '**_'

        # images_dir = os.path.join(exp_dir, 'images')
        # other_dir = os.path.join(exp_dir, 'other_outputs')
        # models_dir = os.path.join(exp_dir, 'trained_models')
        full_mean, full_var = checkTaskMeanVariance(task_name)
        comments = comments + '_taskmean:{}_taskvar:{}'.format(full_mean, full_var)

        ChooseGPU(GPU)
        setReproducible(seed)

        timerepeat = str2val(comments, 'timerepeat', int, default=1)
        maxlen = str2val(comments, 'maxlen', int, default=100)
        comments = str2val(comments, 'maxlen', int, default=maxlen, replace=maxlen)

        # task definition
        if 'onlypretrain' in comments:
            epochs = 0
            steps_per_epoch = 0

        train_task_args = dict(timerepeat=timerepeat, epochs=epochs, batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch,
                               name=task_name, train_val_test='train', maxlen=maxlen, comments=comments)
        gen_train = Task(**train_task_args)

        comments += '_batchsize:' + str(gen_train.batch_size)

        if initializer in esoteric_initializers_list:
            initializer = get_initializer(initializer_name=initializer)

        comments = comments if task_name in language_tasks else comments.replace('embproj', 'simplereadout')
        initial_state = None
        model_args = dict(
            task_name=task_name, net_name=net_name, n_neurons=n_neurons,
            lr=lr, stack=stack, loss_name=loss_name,
            embedding=embedding, optimizer_name=optimizer_name, lr_schedule=lr_schedule,
            weight_decay=weight_decay, clipnorm=clipnorm, initializer=initializer, comments=comments,
            in_len=gen_train.in_len, n_in=gen_train.in_dim, out_len=gen_train.out_len,
            n_out=gen_train.out_dim, final_epochs=gen_train.epochs, seed=seed,
        )

        print('Finding the LSC...')

        new_model_args = copy.deepcopy(model_args)
        new_comments = new_model_args['comments'] + '_reoldspike'

        new_comments = new_comments + '_waddnoise'
        # new_comments = new_comments + '_reducevar'
        new_comments = new_comments + '_randlambda1'

        new_batch_size = batch_size

        lsclr = 7.2e-2  # 3.14e-3 # 7.2e-4 # 7.2e-3 #

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

        # lscw_filepath = os.path.join(models_dir, 'lsc')
        # save_weights_path = lscw_filepath if 'savelscweights' in comments else None

        time_steps = str2val(comments, 'tsteps', int, default=2) if 'test' in comments else None

        print(json.dumps(new_model_args, indent=4, cls=NumpyEncoder))
        lsclr = str2val(comments, 'lsclr', float, default=lsclr)

        del gen_train
        weights, lsc_results = apply_LSC(
            train_task_args=new_task_args, model_args=new_model_args,
            batch_size=new_batch_size, time_steps=time_steps, lr=lsclr
        )

        norms = lsc_results['save_norms']
        keys = list(norms.keys())
        last_batch = np.unique([k[:8] for k in keys])[-1]
        keys = [k for k in keys if last_batch in k]
        keys.sort()

        final_norms = []
        for k in keys:
            # print(k)

            if not norms[k] == [-1] and not norms[k] == []:
                if not 'dec' in k:
                    final_norms.append(norms[k][-1])

        lsc_results['final_norms'] = final_norms
        lsc_results['final_norms_mean'] = np.mean(final_norms)
        lsc_results['final_norms_std'] = np.std(final_norms)

        print('final_norms: ', np.mean(final_norms), np.std(final_norms))
        means_w = [np.mean(w) for w in weights]
        stds_w = [np.std(w) for w in weights]

        mean_path = os.path.join(gradwincrease, f'means_s{seed}_w{n_neurons}.json')
        with open(mean_path, "w") as fp:
            json.dump(means_w, fp)

        std_path = os.path.join(gradwincrease, f'stds_s{seed}_w{n_neurons}.json')
        with open(std_path, "w") as fp:
            json.dump(stds_w, fp)

        results_path = os.path.join(gradwincrease, f'results_s{seed}_w{n_neurons}.json')
        with open(results_path, "w") as fp:
            json.dump(lsc_results, fp)
