import os, shutil, logging, json, copy

from GenericTools.keras_tools.silence_tensorflow import silence_tf

silence_tf()

from sg_design_lif.neural_models.config import default_config

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_CPP_VMODULE"] = "gpu_process_state=10,gpu_cudamallocasync_allocator=10"

tf.compat.v1.enable_eager_execution()

from GenericTools.keras_tools.esoteric_initializers import esoteric_initializers_list, get_initializer
from GenericTools.stay_organized.VeryCustomSacred import CustomExperiment, ChooseGPU
from GenericTools.stay_organized.utils import timeStructured, setReproducible, str2val, NumpyEncoder, save_results
from GenericTools.keras_tools.esoteric_tasks.time_task_redirection import Task, checkTaskMeanVariance, language_tasks
from GenericTools.stay_organized.submit_jobs import dict2iter
from alif_sg.neural_models.recLSC import apply_LSC

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)

ex = CustomExperiment('-inials', base_dir=CDIR, seed=11)
logger = logging.getLogger('inals')


@ex.config
def config():
    # environment properties
    GPU = None
    seed = 41

    # task and net
    # heidelberg wordptb sl_mnist
    task_name = 'heidelberg'

    # test configuration
    epochs = 2
    steps_per_epoch = 2
    batch_size = None

    # net
    net_name = 'maLSNNb' # LSTM maLSNNb
    stack = '4:3'
    n_neurons = 3

    embedding = 'learned:None:None:{}'.format(n_neurons) if task_name in language_tasks else False
    comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_radius_test'
    # 'truersplit'

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
        stack, batch_size, embedding, n_neurons, lr, task_name, net_name, setting='LSC'
    )

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
    lscrec = bool(str2val(comments, 'lscrec', int, default=1))
    lscdepth = bool(str2val(comments, 'lscdepth', int, default=0))
    lscout = bool(str2val(comments, 'lscout', int, default=0))
    lscin = bool(str2val(comments, 'lscin', int, default=0))
    naswot = str2val(comments, 'naswot', int, default=0)

    if 'allns' in comments:
        lscrec, lscdepth, lscout, lscin = True, True, True, True

    new_model_args = copy.deepcopy(model_args)
    new_comments = new_model_args['comments'] + '_reoldspike'
    new_batch_size = batch_size

    if 'ptb' in task_name:
        new_batch_size = 4
        new_comments = str2val(new_comments, 'batchsize', replace=new_batch_size)

    if 'heidelberg' in task_name and 'maLSNN' in net_name:
        new_batch_size = 100
        if stack in [7, 5, 3]:
            new_batch_size = 32

        new_comments = str2val(new_comments, 'batchsize', replace=new_batch_size)

    new_model_args['comments'] = new_comments
    new_task_args = copy.deepcopy(train_task_args)
    new_task_args['batch_size'] = new_batch_size

    _, lsc_results = apply_LSC(
        train_task_args=new_task_args, model_args=new_model_args, norm_pow=2, n_samples=-1,
        batch_size=new_batch_size,
        rec_norm=lscrec, depth_norm=lscdepth, decoder_norm=lscout, encoder_norm=lscin, naswot=naswot,
        learn=False, steps_per_epoch=1, time_steps=4,
    )

    string_result = json.dumps(lsc_results, indent=4, cls=NumpyEncoder)
    print(string_result)
    save_norms = lsc_results['save_norms']

    # plot using matplotlib
    fig, ax = plt.subplots()
    for key, value in save_norms.items():
        ax.plot(value, label=key)
    # ax.plot(save_norms)
    ax.set(xlabel='time (s)', ylabel='norm')
    plt.legend()

    fig.savefig(images_dir + '/LSC.png')
    plt.close(fig)

    save_results(other_dir, lsc_results)



